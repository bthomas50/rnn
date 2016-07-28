
local TransformedMnistPairs, MnistPairs = torch.class("dp.TransformedMnistPairs", "dp.MnistPairs")

TransformedMnistPairs.isMnist = true

TransformedMnistPairs._name = 'mnist'
TransformedMnistPairs._image_size = {2, 28, 28, 1}
TransformedMnistPairs._image_axes = 'bphwc'
TransformedMnistPairs._feature_size = 1*28*28*2
TransformedMnistPairs._classes = {0, 1}


function TransformedMnistPairs:__init(config)
  self._transformer = config.transformer
  MnistPairs.__init(self, config)
end

--Creates an Mnist DataSet out of inputs, and which_set
function TransformedMnistPairs:createDataSet(inputs, which_set)
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
  local instances, targets = self._transformer:generate(inputs)
  
  local input_view, target_view = dp.ImagePairView(), dp.ClassView()
  input_view:forward(self._image_axes, instances)
  target_view:forward('b', targets)
  target_view:setClasses(self._classes)
  -- construct dataset
  local ds = dp.DataSet{inputs=input_view, targets=target_view, which_set=which_set}
  ds:ioShapes('bphwc', 'b')
  return ds
end