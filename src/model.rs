use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, loss::cross_entropy, ops::softmax_last_dim, Embedding,
    Linear, Optimizer, VarBuilder, VarMap,
};
use rand::{distributions::Distribution, Rng, SeedableRng};
use utils::masked_fill;

const BATCH_SIZE: usize = 32;
const BLOCK_SIZE: usize = 8;
const N_EMBD: usize = 384;
const SEED: u64 = 13345457;
const MAX_ITERS: usize = 3000;
const EVAL_INTERVAL: usize = 300;
const EVAL_ITERS: usize = 200;
const LEARNING_RATE: f64 = 1e-3;

/// One head of self-attention
struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
}

impl Head {
    pub fn new(head_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            key: linear_no_bias(N_EMBD, head_size, vb.pp("key"))?,
            query: linear_no_bias(N_EMBD, head_size, vb.pp("query"))?,
            value: linear_no_bias(N_EMBD, head_size, vb.pp("value"))?,
        })
    }
}

impl Module for Head {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        let (b, t, c) = xs.dims3()?;
        let k = xs.apply(&self.key)?;
        let q = xs.apply(&self.query)?;
        let tril = Tensor::tril2(BLOCK_SIZE, DType::F32, device)?.broadcast_left(b)?;
        let attn_w = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * (1. / (c as f64).sqrt()))?;
        let attn_w = masked_fill(&attn_w, f32::NEG_INFINITY, &tril.eq(0.)?.i((.., ..t, ..t))?)?;
        let attn_w = softmax_last_dim(&attn_w)?;
        let v = xs.apply(&self.value)?;
        attn_w.matmul(&v)
    }
}

struct BigramLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    sa_head: Head,
    lm_head: Linear,
}

impl BigramLanguageModel {
    fn new(vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let token_embedding_table = embedding(vocab_size, N_EMBD, vb.pp("token_embd"))?;
        let position_embedding_table = embedding(BLOCK_SIZE, N_EMBD, vb.pp("pos_embd"))?;
        let sa_head = Head::new(N_EMBD, vb.pp("head"))?;
        let lm_head = linear(N_EMBD, vocab_size, vb)?;
        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            sa_head,
            lm_head,
        })
    }

    fn forward_with_loss(&self, xs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = xs.apply(self)?;
        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b * t, c))?;
        let targets = targets.reshape(b * t)?;

        let loss = cross_entropy(&logits, &targets)?;

        Ok((logits, loss))
    }

    fn generate(&self, xs: &Tensor, max_new_tokens: usize, device: &Device) -> Result<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
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
            let w = &p.to_vec2::<f32>()?[0];
            let distr = rand::distributions::WeightedIndex::new(w).map_err(Error::wrap)?;
            let next_token = distr.sample(&mut rng) as u32;

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
        let x = x.apply(&self.sa_head)?; //apply one head of self attention. (B,T,C)
        x.apply(&self.lm_head) // (B,T,vocab_size)
    }
}

fn encode(s: &str, map: &std::collections::BTreeMap<char, u32>) -> Vec<u32> {
    s.chars().map(|c| *map.get(&c).unwrap()).collect()
}

fn decode(i: &[u32], reverse_map: &std::collections::BTreeMap<u32, char>) -> String {
    i.iter().map(|c| *reverse_map.get(c).unwrap()).collect()
}

fn get_batch(data: &[u32], batch_size: usize, block_size: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    assert!(data.len() >= block_size);
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let mut xx = vec![vec![0u32; block_size]; batch_size];
    let mut yy = vec![vec![0u32; block_size]; batch_size];

    for (_batch_index, (x, y)) in xx.iter_mut().zip(yy.iter_mut()).enumerate() {
        let start = rng.gen_range(0..data.len() - block_size);

        x.copy_from_slice(&data[start..start + block_size]);
        y.copy_from_slice(&data[start + 1..start + block_size + 1]);
    }

    (xx, yy)
}

fn estimate_loss(
    data: (&[u32], &[u32]),
    model: &BigramLanguageModel,
    device: &Device,
) -> Result<(f32, f32)> {
    let (train, val) = data;

    fn process_batch(
        model: &BigramLanguageModel,
        train_data: &[u32],
        val_data: &[u32],
        device: &Device,
    ) -> Result<(f32, f32)> {
        let (xb_train, yb_train) = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE);
        let (xb_val, yb_val) = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE);

        let xb_train = Tensor::new(xb_train, device)?;
        let yb_train = Tensor::new(yb_train, device)?;
        let xb_val = Tensor::new(xb_val, device)?;
        let yb_val = Tensor::new(yb_val, device)?;

        let (_, train_loss) = model.forward_with_loss(&xb_train, &yb_train)?;
        let (_, val_loss) = model.forward_with_loss(&xb_val, &yb_val)?;

        Ok((train_loss.to_scalar::<f32>()?, val_loss.to_scalar::<f32>()?))
    }

    let train_losses: Vec<(f32, f32)> = (0..EVAL_ITERS)
        .map(|_| process_batch(model, train, val, device))
        .collect::<Result<Vec<(f32, f32)>>>()?;

    let train_loss_mean = Tensor::new(
        train_losses
            .iter()
            .map(|(train_loss, _)| *train_loss)
            .collect::<Vec<f32>>(),
        device,
    )?
    .mean(D::Minus1)?
    .to_scalar::<f32>()?;

    let val_loss_mean = Tensor::new(
        train_losses
            .iter()
            .map(|(_, val_loss)| *val_loss)
            .collect::<Vec<f32>>(),
        device,
    )?
    .mean(D::Minus1)?
    .to_scalar::<f32>()?;

    Ok((train_loss_mean, val_loss_mean))
}

fn main() -> Result<()> {
    let device: &Device = &Device::new_metal(0)?;

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

    // get train data
    let (xb, yb) = get_batch(&train, BATCH_SIZE, BLOCK_SIZE);
    println!("inputs: {xb:?}");
    println!("targets: {yb:?}");

    // create a new BigramLanguageModel
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let m = BigramLanguageModel::new(vocab_size, vb)?;

    // run forward pass on the untrained model
    let (logits, loss) = m.forward_with_loss(
        &Tensor::new(xb.clone(), device)?,
        &Tensor::new(yb.clone(), device)?,
    )?;

    println!("logits: {logits}");
    println!("loss: {loss}");

    let (train_loss, val_loss) = estimate_loss((&train, &valid), &m, device)?;
    println!("train loss: {train_loss}, val loss: {val_loss}");
    let xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&xs, 100, device)?;
    println!("generated {g:#?}");
    let g = decode(&g.to_vec2::<u32>()?[0], &reverse_map);
    println!("decoded {g:#?}");

    // train the model
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;
    for iter in 0..MAX_ITERS {
        if iter % EVAL_INTERVAL == 0 {
            let (train_loss, val_loss) = estimate_loss((&train, &valid), &m, device)?;
            println!("iter: {iter}, train loss: {train_loss}, val loss: {val_loss}");
        }

        let (xb, yb) = get_batch(&train, BATCH_SIZE, BLOCK_SIZE);
        let xb = Tensor::new(xb, device)?;
        let yb = Tensor::new(yb, device)?;
        let (_, loss) = m.forward_with_loss(&xb, &yb)?;
        optimizer.backward_step(&loss)?;
    }

    // run forward pass on the trained model
    let mut xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&mut xs, 500, device)?;
    println!("generated {g:#?}");
    let g = decode(&g.to_vec2::<u32>()?[0], &reverse_map);
    print!("decoded {g}");

    Ok(())
}
