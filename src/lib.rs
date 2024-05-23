use candle::{Result, Tensor};

pub fn masked_fill(on_false: &Tensor, on_true: f32, mask: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}
