require 'dp'
require 'rnn'
require 'optim'
require './helpers/Drawing'
require './helpers/CoordinateTransforms'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--evalTest', false, 'model was saved with cuda')
cmd:option('--stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('--overwrite', false, 'overwrite checkpoint')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)
model = xp:model().module
tester = xp:tester() or xp:validator() -- dp.Evaluator
tester:sampler()._epoch_size = nil
conf = tester:feedback() -- dp.Confusion
cm = conf._cm -- optim.ConfusionMatrix

print("Last evaluation of "..(xp:tester() and 'test' or 'valid').." set :")
print(cm)

ds = dp.MnistPairs()

ra = model:findModules('nn.RecurrentAttention')[1]
sg = model:findModules('nn.SpatialGlimpse')[1]

-- stochastic or deterministic
for i=1,#ra.actions do
   local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
   rn.stochastic = opt.stochastic
end

if opt.evalTest then
   conf:reset()
   tester:propagateEpoch(ds:testSet())

   print((opt.stochastic and "Stochastic" or "Deterministic") .. "evaluation of test set :")
   print(cm)
end

inputs = ds:get('test','inputs')
targets = ds:get('test','targets', 'b')

input = inputs:narrow(1,1,10)

model:training() -- otherwise the rnn doesn't save intermediate time-step states
if not opt.stochastic then
   for i=1,#ra.actions do
      local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
      rn.stdev = 0 -- deterministic
   end
end
output = model:forward(input)

locations = ra.actions

input = nn.Convert(ds:ioShapes(),'bpchw'):forward(input)
glimpses = {}
patches = {}

params = nil
for k,location in ipairs(locations) do
  glimpses[k] = {}
  patches[k] = {}
  for j = 1, 2 do
    glimpses[k][j] = {}
    patches[k][j] = {}
  end
end

for i=1,input:size(1) do
  local images = input[i]
  for k,location in ipairs(locations) do
    for j=1, 2 do
      local glimpse = glimpses[k][j]
      local patch = patches[k][j]
      local img = images:select(1,j)
      local xy = location[i]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      local imageX, imageY = locatorXYToImageXY(x, y, ds)

      local gimg = img:clone()
      for d=1,sg.depth do
         local size = sg.height*(sg.scale^(d-1))
         local bbox = {imageY-size/2, imageX-size/2, size, size}
         drawBox(gimg, bbox, 1)
      end
      glimpse[i] = gimg
      local sg_, ps
      if k == 1 then
         sg_ = ra.rnn.initialModule:findModules('nn.SpatialGlimpse')[1]
      else
         sg_ = ra.rnn.sharedClones[k]:findModules('nn.SpatialGlimpse')[1]
      end
      -- print(img:size())
      -- print(sg_.output[i]:size())
      patch[i] = image.scale(img:clone():float(), sg_.output[i]:narrow(1,1,1):float())

      collectgarbage()
    end
  end
end

paths.mkdir('glimpse')

for j, glimpse_pairs in ipairs(glimpses) do
  local glimpse_to_save = torch.cat(glimpse_pairs[1][1], glimpse_pairs[2][1], 2)
  local patches_to_save = torch.cat(patches[j][1][1], patches[j][2][1], 2)
  for k = 2, 10 do
    local glimpse_pair_to_save = torch.cat(glimpse_pairs[1][k], glimpse_pairs[2][k], 2)
    local patch_pairs_to_save = torch.cat(patches[j][1][k], patches[j][2][k], 2)

    glimpse_to_save = torch.cat(glimpse_to_save, glimpse_pair_to_save, 3)
    patches_to_save = torch.cat(patches_to_save, patch_pairs_to_save, 3)
  end
  image.save("glimpse/glimpse"..j..".png", glimpse_to_save)
  image.save("glimpse/patches"..j..".png", patches_to_save)
end
