require 'sys'
require 'cunn'
-- require 'ccn2'

dofile 'TemporalConvolutionMM.lua'

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

string.lpad = function(str, len, char)
   -- pad str with ' ' up to len
   if char == nil then char = ' ' end
   return str .. string.rep(char, len - #str)
end

steps = 4 -- nb of steps in loop to average perf
ops = 2 -- ops per point

runs = {
   {
      -- first layer
      ni = 1,
      no = 64,
      kw = 6,
      iw = 8640,
      bs = 128,
      dw = 6,
   },
   {
      -- second layer
      ni = 64,
      no = 64,
      kw = 6,
      iw = 8640/6,
      bs = 128,
      dw = 1,
   },
   {
      -- 2nd layer of 2 weeks of 1 min data
      ni = 64,
      no = 64,
      kw = 6,
      iw = 24*60*14,
      bs = 16,
      dw = 1,
   },
   {
      -- third layer
      ni = 128,
      no = 128,
      kw = 9,
      iw = 32,
      bs = 128,
      dw = 1,
   },
   {
      -- fourth layer
      ni = 128,
      no = 128,
      kw = 7,
      iw = 16,
      bs = 128,
      dw = 1,
   },
   {  -- layers with small inputs/kernels, seen at the lower ends of the network
      ni = 384,
      no = 384,
      kw = 3,
      iw = 13,
      bs = 128,
      dw = 1,
   },
}

for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   print('')
   print('CONFIG: input = ' .. iw..' * ker = ' .. ni..'x'..no..'x'..kw .. ' (bs = '..bs..', stride = ' .. dw .. ')')

   tests = {}
   if true then
      h = 1
      tests = {{
	 name = 'nn.SpatialConvolutionMM:cuda',
	 model = nn.SpatialConvolutionMM(ni, no, kw, h, dw, h):cuda(),
	 input = torch.randn(bs, ni, h, iw):cuda(),
      }, {
	 name = 'TemporalConvolutionMM:cuda',
	 model = TemporalConvolutionMM(bs, iw, ni, no, kw, dw):cuda(),
	 input = torch.randn(bs, iw, ni):cuda(),
      }, {
	 name = 'nn.SpatialConvolutionMM',
	 model = nn.SpatialConvolutionMM(ni, no, kw, h, dw, h),
	 input = torch.randn(bs, ni, h, iw),
      }, {
	 name = 'TemporalConvolutionMM',
	 model = TemporalConvolutionMM(bs, iw, ni, no, kw, dw),
	 input = torch.randn(bs, iw, ni),
      }, {
	 name = 'TemporalConvolution',
	 model = TemporalConvolution(bs, iw, ni, no, kw, dw),
	 input = torch.randn(bs, iw, ni),
      }, {
	 name = 'nn.TemporalConvolution',
	 model = nn.TemporalConvolution(ni, no, kw, dw),
	 input = torch.randn(bs, iw, ni),
      }}
   elseif false then
      require 'fbcunn'
      tests = {{
	 model = nn.TemporalConvolutionFB(ni, no, kw, dw):cuda(),
	 input = torch.randn(bs, iw, ni):cuda(),
      }, {
	 model = nn.TemporalConvolution(ni, no, kw, dw),
	 input = torch.randn(bs, iw, ni),
      }}
   elseif false then
      tests = {{
	    model = nn.Linear(iw*ni, no):cuda(),
	    input = torch.randn(bs, iw*ni):cuda(),
      }, {
	    model = nn.Linear(iw*ni, no),
	    input = torch.randn(bs, iw*ni),
      }}
   elseif false then
      tests = {{
	    model = nn.ReLU():cuda(),
	    input = torch.randn(bs, iw*ni):cuda(),
      }, {
	    model = nn.ReLU(),
	    input = torch.randn(bs, iw*ni),
      }}
   elseif false then
      tests = {{
	    model = nn.Tanh():cuda(),
	    input = torch.randn(bs, iw*ni):cuda(),
      }, {
	    model = nn.Tanh(),
	    input = torch.randn(bs, iw*ni),
      }}
   elseif false then
      tests = {{
	    model = nn.LogSoftMax():cuda(),
	    input = torch.randn(bs, iw*ni):cuda(),
      }, {
	    model = nn.LogSoftMax(),
	    input = torch.randn(bs, iw*ni),
      }}
   elseif false then
      tests = {{
	    model = nn.Dropout():cuda(),
	    input = torch.randn(bs, iw*ni):cuda(),
      }, {
	    model = nn.Dropout(),
	    input = torch.randn(bs, iw*ni),
      }}
   else
      tests = {{
	    model = nn.TemporalMaxPooling(kw, dw):cuda(),
	    input = torch.randn(bs, iw, ni):cuda(),
      }, {
	    model = nn.TemporalMaxPooling(kw, dw),
	    input = torch.randn(bs, iw, ni),
      }}
   end

   local methods = {
      forward = function (t)
	 t.out = t.model:forward(t.input)
      end,
      outputOutput = function (t)
	 t.out = t.model:updateOutput(t.input)
      end,
      updateGradient = function (t)
	 t.model:updateGradInput(t.input, t.out)
      end,
      accGradParameters = function (t)
	 t.model:accGradParameters(t.input, t.out)
      end,
   }
   
   for mname, method in pairs(methods) do
      for i = 1,#tests do
	 cutorch.synchronize()
	 sys.tic()
	 for t = 1,steps do
	    method(tests[i])
	 end
	 cutorch.synchronize()
	 local tm = sys.toc()/steps
	 
	 local name = tests[i]['name']
	 if name == nil then
	    name = tostring(i)
	 end
	 print(name:lpad(35)..(mname..'():'):lpad(20)..'(tm = ' .. tm .. ')')
      end
      print('')
   end
end

print('')
