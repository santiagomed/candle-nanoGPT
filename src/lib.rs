use candle::{Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

pub fn masked_fill(on_false: &Tensor, on_true: f32, mask: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

// a custom Embedding creation function to allow for custom `mean` and `stdev` values,
// since the default ones don't match Karpathy's nanogpt implementation.
pub fn embedding(in_size: usize, out_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        candle_nn::Init::Randn {
            mean: 0f64,
            stdev: 0.2f64,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}

// a custom Linear creation function to allow for custom `mean` and `stdev` values,
// since the default ones don't match Karpathy's nanogpt implementation.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let ws = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        candle_nn::Init::Randn {
            mean: 0f64,
            stdev: 0.2f64,
        },
    )?;
    let bs = Some(vb.get_with_hints(out_dim, "bias", candle_nn::Init::Const(0.0))?);
    Ok(Linear::new(ws, bs))
}

// a custom Linear creation function to allow for custom `mean` and `stdev` values,
// since the default ones don't match Karpathy's nanogpt implementation.
pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let ws = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        candle_nn::Init::Randn {
            mean: 0f64,
            stdev: 0.2f64,
        },
    )?;

    Ok(Linear::new(ws, None))
}
