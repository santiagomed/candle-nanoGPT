use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    embedding, loss::cross_entropy, ops::softmax_last_dim, Embedding, Optimizer, VarBuilder, VarMap,
};
use rand::{distributions::Distribution, Rng, SeedableRng};

const BATCH_SIZE: usize = 32;
const BLOCK_SIZE: usize = 8;
const SEED: u64 = 13345457;
const MAX_ITERS: usize = 3000;
const EVAL_INTERVAL: usize = 300;
const EVAL_ITERS: usize = 200;
const LEARNING_RATE: f64 = 1e-3;

struct BigramLanguageModel {
    token_embedding_table: Embedding,
}

impl BigramLanguageModel {
    fn new(vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let token_embedding_table = embedding(vocab_size, vocab_size, vb)?;
        Ok(Self {
            token_embedding_table,
        })
    }

    fn forward_with_loss(&self, xs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = xs.apply(&self.token_embedding_table)?;
        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b * t, c))?;
        let targets = targets.reshape(b * t)?;

        let loss = cross_entropy(&logits, &targets)?;

        Ok((logits, loss))
    }

    fn generate(&self, xs: &Tensor, max_new_tokens: usize, device: &Device) -> Result<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut generated = xs.clone();

        for _ in 0..max_new_tokens {
            let logits = generated.apply(self)?;
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let p = softmax_last_dim(&logits)?;

            // multinomial sampling
            let w = &p.to_vec2::<f32>()?[0];
            let distr = rand::distributions::WeightedIndex::new(w).map_err(Error::wrap)?;
            let next_token = distr.sample(&mut rng) as u32;

            let xs_next = Tensor::full(next_token, 1, device)?.unsqueeze(0)?;
            generated = Tensor::cat(&[generated, xs_next], 1)?;
        }

        Ok(generated)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.token_embedding_table)
    }
}

fn encode(s: &str, map: &std::collections::BTreeMap<char, u32>) -> Vec<u32> {
    s.chars().map(|c| *map.get(&c).unwrap()).collect()
}

fn decode(i: &[u32], reverse_map: &std::collections::BTreeMap<u32, char>) -> String {
    i.iter().map(|c| *reverse_map.get(c).unwrap()).collect()
}

fn get_batch(data: &[u32], batch_size: usize, block_size: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
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
    let mut set = std::collections::BTreeSet::new();
    let file = std::fs::read_to_string("./input.txt").unwrap();
    for c in file.chars() {
        set.insert(c);
    }

    // print the sorted set
    for c in &set {
        print!("{}", c);
    }
    let vocab_size = set.len();
    println!("\nvocab size: {}", vocab_size);

    // create a mapping from characters to integers and a reverse mapping
    let mut map = std::collections::BTreeMap::new();
    for (i, c) in set.into_iter().enumerate() {
        map.insert(c, i as u32);
    }
    let mut reverse_map = std::collections::BTreeMap::new();
    for (c, i) in &map {
        reverse_map.insert(*i, *c);
    }

    // encode the text using the mapping
    let data = encode(&file, &map);

    // split the encoded text into training and validation sets
    let n = (data.len() * 9) / 10;
    let (train, valid) = data.split_at(n);
    let train = train.to_vec();
    let valid = valid.to_vec();

    assert_eq!(data.len(), train.len() + valid.len());

    // get train data
    let (batch_size, block_size) = (4, 8);
    let (xb, yb) = get_batch(&train, batch_size, block_size);
    println!("inputs: {xb:?}");
    println!("targets: {yb:?}");

    for b in 0..batch_size {
        for t in 0..block_size {
            let context = &xb[b][..t + 1];
            let target = yb[b][t];
            println!("when input is {context:?} the target: {target}");
        }
    }

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
