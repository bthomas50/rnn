require('dp')
require('./ImagePairView.lua')

local MnistPairs, Mnist = torch.class("dp.MnistPairs", "dp.Mnist")

MnistPairs.isMnist = true

MnistPairs._name = 'mnist'
MnistPairs._image_size = {2, 28, 28, 1}
MnistPairs._image_axes = 'bphwc'
MnistPairs._feature_size = 1*28*28*2
MnistPairs._classes = {0, 1}


function MnistPairs:__init(config)
  Mnist.__init(self, config)
end

function MnistPairs:loadTrainValid()
  --Data will contain a tensor where each row is an example, and where
  --the last column contains the target class.
  local data = self:loadData(self._train_file, self._download_url)
  -- train
  local start = 1
  local num_train_images = math.floor(data[1]:size(1) * (1-self._valid_ratio))
  self:trainSet( self:createDataSet(data[1]:narrow(1, start, num_train_images), 'train' ) )
  -- valid
  if self._valid_ratio == 0 then
    print"Warning : No Valid Set due to valid_ratio == 0"
    return
  end
  start = num_train_images
  local num_valid_images = data[1]:size(1) - start
  self:validSet( self:createDataSet(data[1]:narrow(1, start, num_valid_images), 'valid') )
  return self:trainSet(), self:validSet()
end

function MnistPairs:loadTest()
  local test_data = self:loadData(self._test_file, self._download_url)
  self:testSet( self:createDataSet(test_data[1], 'test') )
  return self:testSet()
end

function MnistPairs:ioShapes()
  return self._image_axes
end
--Creates an Mnist DataSet out of inputs, and which_set
function MnistPairs:createDataSet(inputs, which_set)
  if self._shuffle then
    local indices = torch.randperm(inputs:size(1)):long()
    inputs = inputs:index(1, indices)
  end
  if self._binarize then
    dp.DataSource.binarize(inputs, 128)
  end
  if self._scale and not self._binarize then
    dp.DataSource.rescale(inputs, self._scale[1], self._scale[2])
  end
  local old_size = inputs:size()
  local better_inputs = inputs.new(old_size[1], 2, old_size[2], old_size[3], old_size[4])
  local targets = inputs.new(old_size[1])
  for i = 1, old_size[1] do
    better_inputs[i][1] = inputs[i]
    targets[i] = (i % 2) + 1
    if targets[i] == 2 then -- if this instance is a match
      better_inputs[i][2] = inputs[i]
    elseif i == old_size[1] then
      better_inputs[i][2] = inputs[1]
    else
      better_inputs[i][2] = inputs[i + 1]
    end
  end
  -- class 0 will have index 1, class 1 index 2.
  
  -- construct inputs and targets dp.Views 
  local input_view, target_view = dp.ImagePairView(), dp.ClassView()
  input_view:forward(self._image_axes, better_inputs)
  target_view:forward('b', targets)
  target_view:setClasses(self._classes)
  -- construct dataset
  local ds = dp.DataSet{inputs=input_view, targets=target_view, which_set=which_set}
  ds:ioShapes('bphwc', 'b')
  return ds
end