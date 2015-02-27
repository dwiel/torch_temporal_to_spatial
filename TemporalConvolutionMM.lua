require 'nn'

function _SpatialTemporalConvolution(width, inp, outp, kw, dw, SpatialConvolution)
   -- make sure that SpatialConvolution is one of the two valid ones
   assert(
      (SpatialConvolution == nn.SpatialConvolution) or
      (SpatialConvolution == nn.SpatialConvolutionMM)
   )

   height=1
   kh=1
   dh=1

   output_width = (width - kw) / dw + 1

   -- batchmode false on Reshape keeps reshape from heuristically
   -- deciding that the first dimension is a batch

   model = nn.Sequential()
   model:add(nn.Transpose({1,2}))
   model:add(nn.Reshape(inp, height, width, false))
   model:add(SpatialConvolution(inp, outp, kw, kh, dw, dh))
   model:add(nn.View(outp, output_width))
   model:add(nn.Transpose({1,2}))

   return model
end

function TemporalConvolutionMM(width, inp, outp, kw, dw)
   -- use nn.SpatialConvolutionMM
   return _SpatialTemporalConvolution(width, inp, outp, kw, dw, nn.SpatialConvolutionMM)
end

function TemporalConvolution(width, inp, outp, kw, dw)
   -- use nn.SpatialConvolution
   return _SpatialTemporalConvolution(width, inp, outp, kw, dw, nn.SpatialConvolution)
end
