use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, ops::softmax_last_dim, Embedding, Optimizer, VarBuilder, VarMap};
use rand::{distributions::Distribution, Rng, SeedableRng};

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
        let logits = self.token_embedding_table.forward(xs)?;
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

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 32;
const SEED: u64 = 0;

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

    for batch_index in 0..BATCH_SIZE {
        let start = rng.gen_range(0..data.len() - BLOCK_SIZE);

        for block_index in 0..BLOCK_SIZE {
            xx[batch_index][block_index] = data[start + block_index];
            yy[batch_index][block_index] = data[start + block_index + 1];
        }
    }

    (xx, yy)
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
    let n = 0.9 * encoded.len() as f64;
    let train = &encoded[0..n as usize];
    let _valid = &encoded[n as usize..];

    // get a batch of training data
    let (xb, yb) = get_batch(train);
    let xb = Tensor::new(&xb, device)?;
    let yb = Tensor::new(&yb, device)?;
    println!("{xb}");
    println!("{yb}");

    // create a new BigramLanguageModel
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let m = BigramLanguageModel::new(vocab_size, vb)?;
    let (logits, loss) = m.forward_with_loss(&xb, &yb)?;

    println!("untrained logits {logits}");
    println!("initial loss {loss}");

    // run forward pass on the untrained model
    let mut xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&mut xs, 100, device)?;
    println!("generated {g:#?}");
    let g = g.to_vec2::<u32>()?[0].clone();
    let g = decode(&g, &reverse_map);
    println!("decoded {g:#?}");

    // train the model
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;
    for _ in 0..10000 {
        let (xb, yb) = get_batch(train);
        let xb = Tensor::new(&xb, device)?;
        let yb = Tensor::new(&yb, device)?;
        let (_, loss) = m.forward_with_loss(&xb, &yb)?;
        optimizer.backward_step(&loss)?;
    }

    // run forward pass on the trained model
    let mut xs = Tensor::zeros((1, 1), DType::U32, device)?;
    let g = m.generate(&mut xs, 100, device)?;
    println!("generated {g:#?}");
    let g = g.to_vec2::<u32>()?[0].clone();
    let g = decode(&g, &reverse_map);
    println!("decoded {g:#?}");

    Ok(())
}
