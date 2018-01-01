(ns cortex-examples.common-networks
  (:require
            [cortex.nn.execute :as execute]
            [cortex.nn.compute-binding :as compute-binding]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.graph :as graph]
            [cortex.nn.network :as network]
            [cortex.nn.layers :as layers]
            [cortex.util :as util]
            [cortex.loss.core :as loss]))

(defn initial-description
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/tanh 100 50)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/tanh 50 30)
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])
