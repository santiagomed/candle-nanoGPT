use std::collections::HashMap;

use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, ops::softmax_last_dim, Embedding, Optimizer, VarBuilder, VarMap};
use rand::{distributions::Distribution, Rng, SeedableRng};

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 32;
const SEED: u64 = 1337;
const MAX_ITERS: usize = 3000;
const EVAL_INTERVAL: usize = 300;
const EVAL_ITERS: usize = 400;
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

        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;

        Ok((logits, loss))
    }

    fn generate(&self, xs: &mut Tensor, max_new_tokens: usize, device: &Device) -> Result<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1345425245u64);
        for _ in 0..max_new_tokens {
            let logits = xs.apply(self)?;
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let p = softmax_last_dim(&logits)?;

            // multinomial sampling
            let w = &p.to_vec2::<f32>()?[0];
            let distr = rand::distributions::WeightedIndex::new(w).map_err(Error::wrap)?;
            let next_token = distr.sample(&mut rng) as u32;

            let xs_next = Tensor::full(next_token, 1, device)?.unsqueeze(0)?;
            *xs = Tensor::cat(&[xs.clone(), xs_next.clone()], 1)?;
        }
        // println!("xs {xs}");
        Ok(xs.clone())
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.token_embedding_table)
    }
}

fn encode(s: &str, map: &std::collections::BTreeMap<char, u32>) -> Vec<u32> {
    let mut encoded = Vec::new();
    for c in s.chars() {
        encoded.push(*map.get(&c).unwrap());
    }
    encoded
}

fn decode(i: &[u32], reverse_map: &std::collections::BTreeMap<u32, char>) -> String {
    let mut decoded = String::new();
    for c in i {
        decoded.push(*reverse_map.get(c).unwrap());
    }
    decoded
}

fn get_batch(
    data: &[u32],
) -> (
    [[u32; BLOCK_SIZE]; BATCH_SIZE],
    [[u32; BLOCK_SIZE]; BATCH_SIZE],
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let mut xx = [[0u32; BLOCK_SIZE]; BATCH_SIZE];
    let mut yy = [[0u32; BLOCK_SIZE]; BATCH_SIZE];

    for (_batch_index, (x, y)) in xx.iter_mut().zip(yy.iter_mut()).enumerate() {
        let start = rng.gen_range(0..data.len() - BLOCK_SIZE);

        x.copy_from_slice(&data[start..start + BLOCK_SIZE]);
        y.copy_from_slice(&data[start + 1..start + BLOCK_SIZE + 1]);
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
        let (xb_train, yb_train) = get_batch(train_data);
        let (xb_val, yb_val) = get_batch(val_data);

        let xb_train = Tensor::new(&xb_train, device)?;
        let yb_train = Tensor::new(&yb_train, device)?;
        let xb_val = Tensor::new(&xb_val, device)?;
        let yb_val = Tensor::new(&yb_val, device)?;

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
    let encoded = encode(&file, &map);

    // split the encoded text into training and validation sets
    let n = (0.9 * encoded.len() as f32) as usize;
    let train = &encoded[0..n];
    let valid = &encoded[n..];
    assert_eq!(encoded.len(), train.len() + valid.len());

    // create a new BigramLanguageModel
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let m = BigramLanguageModel::new(vocab_size, vb)?;

    // run forward pass on the untrained model
    let mut xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&mut xs, 100, device)?;
    println!("generated {g:#?}");
    let g = decode(&g.to_vec2::<u32>()?[0], &reverse_map);
    println!("decoded {g:#?}");

    // train the model
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;
    let mut best_val_loss = f32::INFINITY;
    let mut patience = 10; // number of epochs to wait for improvement;
    for iter in 0..MAX_ITERS {
        if iter % EVAL_INTERVAL == 0 {
            let (train_loss, val_loss) = estimate_loss((train, valid), &m, device)?;
            println!("iter: {iter}, train loss: {train_loss}, val loss: {val_loss}");
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience = 10; // reset patience
            } else {
                patience -= 1;
                if patience == 0 {
                    println!("Early stopping at iter: {iter}");
                    break;
                }
            }
        }

        let (xb, yb) = get_batch(train);
        let xb = Tensor::new(&xb, device)?;
        let yb = Tensor::new(&yb, device)?;
        let (_, loss) = m.forward_with_loss(&xb, &yb)?;
        optimizer.backward_step(&loss)?;
    }

    // run forward pass on the trained model
    let mut xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&mut xs, 500, device)?;
    println!("generated {g:#?}");
    let g = g.to_vec2::<u32>()?[0].clone();
    let g = decode(&g, &reverse_map);
    print!("decoded {g}");

    Ok(())
}
