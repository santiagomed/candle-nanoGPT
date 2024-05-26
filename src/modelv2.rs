use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    layer_norm, loss::cross_entropy, ops::softmax_last_dim, Activation, Dropout, Embedding,
    LayerNorm, LayerNormConfig, Linear, Optimizer, VarBuilder, VarMap,
};
use rand::{distributions::Distribution, rngs::StdRng, Rng, SeedableRng};
use utils::{embedding, linear, linear_no_bias, masked_fill};

const BATCH_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const N_EMBD: usize = 384;
const SEED: u64 = 1337;
const MAX_ITERS: usize = 5000;
const EVAL_INTERVAL: usize = 500;
const EVAL_ITERS: usize = 200;
const N_HEAD: usize = 6;
const N_LAYER: usize = 6;
const LEARNING_RATE: f64 = 3e-4;
const DROPOUT: f32 = 0.2;
const TRAIN: bool = true;

/// One head of self-attention
struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    dropout: Dropout,
}

impl Head {
    pub fn new(head_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            key: linear_no_bias(N_EMBD, head_size, vb.pp("key"))?,
            query: linear_no_bias(N_EMBD, head_size, vb.pp("query"))?,
            value: linear_no_bias(N_EMBD, head_size, vb.pp("value"))?,
            dropout: Dropout::new(DROPOUT),
        })
    }
}

impl Module for Head {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        let (b, t, c) = xs.dims3()?;
        let k = xs.apply(&self.key)?;
        let q = xs.apply(&self.query)?;
        let tril = Tensor::tril2(BLOCK_SIZE, DType::U8, device)?.broadcast_left(b)?;
        let attn_w = (q.matmul(&k.t()?)? * (1. / (c as f64).sqrt()))?;
        let attn_w = masked_fill(&attn_w, f32::NEG_INFINITY, &tril.eq(0.)?.i((.., ..t, ..t))?)?;
        let attn_w = softmax_last_dim(&attn_w)?;
        let attn_w = self.dropout.forward(&attn_w, TRAIN)?;
        let v = xs.apply(&self.value)?;
        attn_w.matmul(&v)
    }
}

struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(n_heads: usize, head_size: usize, vb: VarBuilder) -> Result<Self> {
        let heads = (0..n_heads)
            .map(|_| Head::new(head_size, vb.clone()))
            .collect::<Result<Vec<_>>>()?;
        let proj = linear(n_heads * head_size, N_EMBD, vb.pp("proj"))?;
        let dropout = Dropout::new(DROPOUT);
        Ok(Self {
            heads,
            proj,
            dropout,
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let heads = self
            .heads
            .iter()
            .map(|head| xs.apply(head))
            .collect::<Result<Vec<_>>>()?;
        let xs = Tensor::cat(&heads, D::Minus1)?.apply(&self.proj)?;
        self.dropout.forward(&xs, TRAIN)
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    activation: Activation,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(N_EMBD, 4 * N_EMBD, vb.pp("linear1"))?,
            linear2: linear(4 * N_EMBD, N_EMBD, vb.pp("linear2"))?,
            activation: Activation::Relu,
            dropout: Dropout::new(DROPOUT),
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.dropout.forward(
            &xs.apply(&self.linear1)?
                .apply(&self.activation)?
                .apply(&self.linear2)?,
            TRAIN,
        )
    }
}

struct Block {
    sa_heads: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl Block {
    pub fn new(num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_size = N_EMBD / num_heads;
        let sa_heads = MultiHeadAttention::new(num_heads, head_size, vb.pp("sa_heads"))?;
        let ffwd = FeedForward::new(vb.pp("ffwd"))?;
        let ln1 = layer_norm(N_EMBD, LayerNormConfig::default(), vb.pp("ln1"))?;
        let ln2 = layer_norm(N_EMBD, LayerNormConfig::default(), vb.pp("ln2"))?;
        Ok(Self {
            sa_heads,
            ffwd,
            ln1,
            ln2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.add(&xs.apply(&self.ln1)?.apply(&self.sa_heads)?)?;
        xs.add(&xs.apply(&self.ln2)?.apply(&self.ffwd)?)
    }
}

struct BigramLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    ln: LayerNorm,
}

impl BigramLanguageModel {
    fn new(vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let token_embedding_table = embedding(vocab_size, N_EMBD, vb.pp("token_embd"))?;
        let position_embedding_table = embedding(BLOCK_SIZE, N_EMBD, vb.pp("pos_embd"))?;
        let blocks = (0..N_LAYER)
            .map(|i| Block::new(N_HEAD, vb.pp(format!("blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let lm_head = linear(N_EMBD, vocab_size, vb.pp("lm_head"))?;
        let ln = layer_norm(N_EMBD, LayerNormConfig::default(), vb.pp("ln"))?;

        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            blocks,
            lm_head,
            ln,
        })
    }

    fn evaluate_loss(&self, xs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = xs.apply(self)?;
        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b * t, c))?;
        let targets = targets.reshape(b * t)?;
        let loss = cross_entropy(&logits, &targets)?;
        Ok((logits, loss))
    }

    fn generate(
        &self,
        xs: &Tensor,
        max_new_tokens: usize,
        rng: &mut StdRng,
        device: &Device,
    ) -> Result<Tensor> {
        let mut xs = xs.clone(); // (B, T) array of indices in the current context

        for _ in 0..max_new_tokens {
            // crop xs to the last block_size of tokens
            let xs_cond = if xs.dim(1)? > BLOCK_SIZE {
                xs.i((.., xs.dim(1)? - BLOCK_SIZE..))?
            } else {
                xs.i((.., ..))?
            };
            // get the predictions
            let logits = xs_cond.apply(self)?;
            // focus only on the last time step
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            // apply softmax to get the probabilities
            let p = softmax_last_dim(&logits)?;

            // multinomial sampling
            let distr = rand::distributions::WeightedIndex::new(&p.to_vec2::<f32>()?[0])
                .map_err(Error::wrap)?;
            let next_token = distr.sample(rng) as u32;

            // append sample index to the running sequence
            let xs_next = Tensor::full(next_token, 1, device)?.unsqueeze(0)?; // (B=1, 1) [[next_token]]
            xs = Tensor::cat(&[xs, xs_next], 1)?; // (B, T+1)
        }

        Ok(xs)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        let (_b, t) = xs.dims2()?;
        let tok_emb = xs.apply(&self.token_embedding_table)?; // (B,T,C)
        let pos_emb =
            Tensor::arange(0u32, t as u32, device)?.apply(&self.position_embedding_table)?; // (T,C)
        let x = tok_emb.broadcast_add(&pos_emb)?; // (B,T,C)
        let x = self
            .blocks
            .iter()
            .try_fold(x, |x, block| x.apply(block))?
            .apply(&self.ln)?; // (B,T,C)
        x.apply(&self.lm_head) // (B,T,vocab_size)
    }
}

fn encode(s: &str, map: &std::collections::BTreeMap<char, u32>) -> Vec<u32> {
    s.chars().map(|c| *map.get(&c).unwrap()).collect()
}

fn decode(i: &[u32], reverse_map: &std::collections::BTreeMap<u32, char>) -> String {
    i.iter().map(|c| *reverse_map.get(c).unwrap()).collect()
}

fn get_batch(
    data: &[u32],
    batch_size: usize,
    block_size: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    assert!(data.len() >= block_size);

    let mut xx = vec![vec![0u32; block_size]; batch_size];
    let mut yy = vec![vec![0u32; block_size]; batch_size];

    for (_batch_index, (x, y)) in xx.iter_mut().zip(yy.iter_mut()).enumerate() {
        let start = rng.gen_range(0..data.len() - block_size - 1);

        x.copy_from_slice(&data[start..start + block_size]);
        y.copy_from_slice(&data[start + 1..start + block_size + 1]);
    }

    (xx, yy)
}

fn estimate_loss(
    data: (&[u32], &[u32]),
    model: &BigramLanguageModel,
    rng: &mut StdRng,
    device: &Device,
) -> Result<(f32, f32)> {
    let (train, val) = data;

    fn process_batch(
        model: &BigramLanguageModel,
        train_data: &[u32],
        val_data: &[u32],
        rng: &mut StdRng,
        device: &Device,
    ) -> Result<(f32, f32)> {
        let (xb_train, yb_train) = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, rng);
        let (xb_val, yb_val) = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, rng);

        let xb_train = Tensor::new(xb_train, device)?;
        let yb_train = Tensor::new(yb_train, device)?;
        let xb_val = Tensor::new(xb_val, device)?;
        let yb_val = Tensor::new(yb_val, device)?;

        let (_, train_loss) = model.evaluate_loss(&xb_train, &yb_train)?;
        let (_, val_loss) = model.evaluate_loss(&xb_val, &yb_val)?;

        Ok((train_loss.to_scalar::<f32>()?, val_loss.to_scalar::<f32>()?))
    }

    let train_losses: Vec<(f32, f32)> = (0..EVAL_ITERS)
        .map(|_| process_batch(model, train, val, rng, device))
        .collect::<Result<Vec<(f32, f32)>>>()?;

    let train_loss_mean = Tensor::new(
        train_losses
            .iter()
            .map(|(train_loss, _)| *train_loss)
            .collect::<Vec<f32>>(),
        device,
    )?
    .mean(0)?
    .to_scalar::<f32>()?;

    let val_loss_mean = Tensor::new(
        train_losses
            .iter()
            .map(|(_, val_loss)| *val_loss)
            .collect::<Vec<f32>>(),
        device,
    )?
    .mean(0)?
    .to_scalar::<f32>()?;

    Ok((train_loss_mean, val_loss_mean))
}

fn main() -> Result<()> {
    let device: &Device = &Device::new_cuda(0)?;

    // read from file and create a sorted set of all unique characters in the text
    let file = std::fs::read_to_string("./input.txt").unwrap();
    let set = file
        .chars()
        .map(|c| c)
        .collect::<std::collections::BTreeSet<_>>();
    let vocab_size = set.len();

    // create a mapping from characters to integers and a reverse mapping
    let map = set
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i as u32))
        .collect::<std::collections::BTreeMap<_, _>>();
    let reverse_map = map
        .iter()
        .map(|(c, i)| (*i, *c))
        .collect::<std::collections::BTreeMap<_, _>>();

    // encode the text using the mapping
    let data = encode(&file, &map);

    // split the encoded text into training and validation sets
    let n = (data.len() * 9) / 10;
    let (train, valid) = data.split_at(n);
    let train = train.to_vec();
    let valid = valid.to_vec();

    assert_eq!(data.len(), train.len() + valid.len());

    // create a random number generator
    let mut rng = StdRng::seed_from_u64(SEED);

    // get train data
    let (xb, yb) = get_batch(&train, BATCH_SIZE, BLOCK_SIZE, &mut rng);
    println!("inputs: {xb:?}");
    println!("targets: {yb:?}");

    // create a new BigramLanguageModel
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let m = BigramLanguageModel::new(vocab_size, vb)?;

    // run forward pass on the untrained model
    let (logits, loss) = m.evaluate_loss(
        &Tensor::new(xb.clone(), device)?,
        &Tensor::new(yb.clone(), device)?,
    )?;

    println!("logits: {logits}");
    println!("loss: {loss}");

    let (train_loss, val_loss) = estimate_loss((&train, &valid), &m, &mut rng, device)?;
    println!("train loss: {train_loss}, val loss: {val_loss}");
    let xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&xs, 100, &mut rng, device)?;
    println!("generated {g:#?}");
    let g = decode(&g.to_vec2::<u32>()?[0], &reverse_map);
    println!("decoded {g:#?}");

    // train the model
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;
    for iter in 0..MAX_ITERS {
        if iter % EVAL_INTERVAL == 0 {
            let (train_loss, val_loss) = estimate_loss((&train, &valid), &m, &mut rng, device)?;
            println!("iter: {iter}, train loss: {train_loss}, val loss: {val_loss}");
        }

        let (xb, yb) = get_batch(&train, BATCH_SIZE, BLOCK_SIZE, &mut rng);
        let xb = Tensor::new(xb, device)?;
        let yb = Tensor::new(yb, device)?;
        let (_, loss) = m.evaluate_loss(&xb, &yb)?;
        optimizer.backward_step(&loss)?;
    }

    // run forward pass on the trained model
    let mut xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&mut xs, 500, &mut rng, device)?;
    println!("generated {g:#?}");
    let g = decode(&g.to_vec2::<u32>()?[0], &reverse_map);
    print!("decoded {g}");

    Ok(())
}
