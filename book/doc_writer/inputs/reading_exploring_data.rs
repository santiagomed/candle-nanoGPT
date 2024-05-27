use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor, D};

fn main() -> Result<()> {
    let device: &Device = &Device::Cpu;

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
    Ok(())
}
