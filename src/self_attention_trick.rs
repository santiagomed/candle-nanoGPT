/// toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{ops::softmax_last_dim, VarBuilder, VarMap};
use utils::masked_fill;

fn allclose(a: &Tensor, b: &Tensor, rtol: f32, atol: f32, device: &Device) -> Result<bool> {
    let diff = a.sub(&b)?;
    let abs_diff = diff.abs()?;
    let abs_a = a.abs()?;
    let abs_b = b.abs()?;
    let rtol = abs_b.broadcast_mul(&Tensor::new(&[rtol + atol], device)?)?;
    let ones_mask = abs_diff.le(&rtol)?.eq(1.)?;
    let ok = ones_mask.sum_all()?.to_scalar::<u8>()? == ones_mask.elem_count() as u8;
    if !ok {
        println!("max_abs_diff: {}", abs_diff);
        println!("max_abs_a: {}", abs_a);
        println!("max_abs_b: {}", abs_b);
        println!("rtol: {}", rtol);
    }
    Ok(ok)
}

fn main() -> Result<()> {
    let device: &Device = &Device::new_metal(0)?;
    let a = Tensor::tril2(3, DType::F32, device)?;
    let sum = a.sum_keepdim(1)?;
    let a = a.broadcast_div(&sum)?;
    let b = Tensor::rand(0f32, 10f32, (3, 2), device)?
        .to_dtype(DType::I64)?
        .to_dtype(DType::F32)?;
    println!("b shape: {b:?}");
    let c = a.matmul(&b)?;
    // print each tensor as follows: println!("a={}", a)
    println!("a={}", a);
    println!("---");
    println!("b={}", b);
    println!("---");
    println!("c={}", c);

    // Version #1: using a loop
    let (b, t, c) = (4, 8, 2);
    let x = Tensor::randn(0f32, 1f32, (b, t, c), device)?;

    let mut xbow = vec![vec![vec![0f32; c]; t]; b];
    for b in 0..b {
        for t in 0..t {
            let xprev = x.i((b, 0..t + 1))?;
            xbow[b][t] = xprev.mean(0)?.to_vec1()?;
        }
    }

    let xbow = Tensor::new(xbow, device)?;
    println!("xbow={xbow}");

    // Version #2: using matrix multiply for a weighted aggregation
    let wei = Tensor::tril2(t, DType::F32, device)?;
    let sum = wei.sum_keepdim(1)?;
    let wei = wei.broadcast_div(&sum)?;
    let xbow2 = wei.broadcast_matmul(&x)?;
    assert!(allclose(&xbow, &xbow2, 1e-8, 1e-5, device)?);

    // Version #3: use Softmax
    let tril = Tensor::tril2(t, DType::F32, device)?;
    let wei = Tensor::zeros_like(&tril)?;
    let wei = masked_fill(&wei, f32::NEG_INFINITY, &tril.eq(0.)?)?;
    let wei = softmax_last_dim(&wei)?;
    let xbow3 = wei.broadcast_matmul(&x)?;
    assert!(allclose(&xbow, &xbow3, 1e-8, 1e-5, device)?);

    // Version #4: self-attention
    let (b, t, c) = (4, 8, 32);
    let x = Tensor::randn(0f32, 1f32, (b, t, c), device)?;
    println!("x={x}");

    let head_size = 16;
    let key = candle_nn::linear_no_bias(
        c,
        head_size,
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, device),
    )?;
    let query = candle_nn::linear_no_bias(
        c,
        head_size,
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, device),
    )?;
    let value = candle_nn::linear_no_bias(
        c,
        head_size,
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, device),
    )?;
    let k = x.apply(&key)?.transpose(1, 2)?; // (B, T, 16)
    let q = x.apply(&query)?; // (B, T, 16)
    let wei = q.matmul(&k)?; // (B, T, 16) x (B, 16, T) = (B, T, T)
    println!("wei={wei}");

    let tril = Tensor::tril2(t, DType::F32, device)?.broadcast_left(b)?;
    let wei = masked_fill(&wei, f32::NEG_INFINITY, &tril.eq(0.)?)?;
    let wei = softmax_last_dim(&wei)?;
    println!("wei={}", wei.i(0)?);
    let v = x.apply(&value)?;
    let out = wei.matmul(&v)?;
    println!("out={out:?}");

    // Scaled dot-product attention
    let k = Tensor::randn(0f32, 1f32, (b, t, head_size), device)?;
    let q = Tensor::randn(0f32, 1f32, (b, t, head_size), device)?;
    let wei = (q.matmul(&k.transpose(1, 2)?)? * (1. / (head_size as f64).sqrt()))?;
    println!("k var={}", k.var(0)?);
    println!("q var={}", q.var(0)?);
    println!("w var={}", wei.var(0)?);

    println!(
        "{}",
        softmax_last_dim(&Tensor::new(
            &[0.1f32, -0.2f32, 0.3f32, -0.2f32, 0.5f32],
            device
        )?)?,
    );
    println!(
        "{}",
        softmax_last_dim(
            &(Tensor::new(&[0.1f32, -0.2f32, 0.3f32, -0.2f32, 0.5f32], device)? * 8.)?
        )?,
    );

    Ok(())
}
