(ns cortex-examples.common-networks
  (:require
            [cortex.nn.execute :as execute]
            [cortex.nn.compute-binding :as compute-binding]
            [cortex.experiment.util :as experiment-util]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.graph :as graph]
            [cortex.util :as util]
            [cortex.nn.network :as network]
            [cortex.nn.layers :as layers]
            [cortex.util :as util]
            [mnist-classification.core :as mcore]
            [cortex.experiment.train :as etrain]
            [confuse.binary-class-metrics :as bcm]
            [cortex.loss.core :as loss]))

(defn initial-description
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/prelu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/prelu)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn initial-description-mnist
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 1000)
   (layers/relu :center-loss {:label-indexes {:stream :labels}
                              :label-inverse-counts {:stream :labels}
                              :labels {:stream :labels}
                              :alpha 0.9
                              :lambda 1e-4})
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn initial-description-mnist-wocl
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 1000)
   (layers/relu)
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn lenet-mnist-wocl
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 2 1 32)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 2 1 64)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 2 1 128)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 100)
   (layers/relu)
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn train-forever
  ([inet filename ]
   (mcore/ensure-images-on-disk!)
   (println "Training forever.")
   (let [training-folder (str mcore/dataset-folder "training")
         test-folder (str mcore/dataset-folder "test")
         [train-ds test-ds] [(-> training-folder
                                 (experiment-util/create-dataset-from-folder mcore/class-mapping)
                                 (experiment-util/infinite-class-balanced-dataset))
                             (-> test-folder
                                 (experiment-util/create-dataset-from-folder mcore/class-mapping))]]
     (etrain/train-n
      inet 
      train-ds test-ds
      :epoch-count 25
      :network-filestem filename))))

(comment 
  (def res (mapv train-forever
                 [(initial-description-mnist 28 28 10)
                  (initial-description-mnist-wocl 28 28 10)]
                 ["mnist-with-cenloss" "mnist-wo-cenloss"])))
;;with cen-loss- best loss is 0.083
;;witout cen-los best loss is 0.100
;;(def res2 (mapv train-forever [(lenet-mnist-wocl 28 28 10)]))
;0.75 best loss with initial-description
;(def res-wol2 (train-forever))

(def mnist-cenloss (util/read-nippy-file "mnist-with-cenloss.nippy"))
(def mnist-wo-cenloss (util/read-nippy-file "mnist-wo-cenloss.nippy"))

;(-> mnist-cenloss :cv-loss)
;(-> mnist-wo-cenloss :cv-loss)
(defn get-2d
  [mnist-network train? ]
  (let [test-folder (str mcore/dataset-folder (if (true? train?) "training" "test"))
        test-ds (-> test-folder
                    (experiment-util/create-dataset-from-folder mcore/class-mapping)
                                        ;(experiment-util/infinite-class-balanced-dataset)
                    shuffle
                    )
        network-bottleneck (network/dissoc-layers-from-network mnist-network :linear-2)
        layers-to-add (->
                       ;;[(layers/linear 2 :weight-initialization-type :constant)]
                        ;;[(layers/linear 2 :weight-initialization-type :xavier)]
                       [(layers/linear 2 :weight-initialization-type :constant)
                        (layers/linear num-classes)
                        (layers/softmax :id :labels) ]
                          flatten vec)
        modified-network (network/assoc-layers-to-network network-bottleneck layers-to-add)
        actual-labels (->> test-ds (mapv (comp util/max-index :labels )))]
    (->> test-ds
         (take 1000)
         (execute/run modified-network )
         (mapv :linear-2)
         (mapv into (mapv vector actual-labels))
         )
    ))

(def k1 (get-2d mnist-cenloss false))
(def k1 (get-2d mnist-cenloss true))
(->> k1 (mapv first ) set)
;(get-in (vec [(layers/linear 2)]) [0])
;(assoc-in (-> [(layers/linear 2)] flatten vec) [0 :parents] 10)
;(-> res :compute-graph :edges)
