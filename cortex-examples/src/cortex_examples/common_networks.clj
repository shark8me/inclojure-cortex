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
            [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as rm]
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

(defn mnist-relu
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear->relu 1000)
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn mnist-tanh
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear->logistic 100 :weights (m/reshape (mapv #(- 400 %)
                                                          (rm/sample-rand-int (* 100 800) 400))
                                                    [100 800]))
   (layers/linear->logistic 50)
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

(defn mnist-tanh
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/linear->logistic 100 :weights (m/reshape (mapv #(- 400 %)
                                                          (rm/sample-rand-int (* 100 (* 28 28)) 800))
                                                    [100 (* 28 28)]))
   (layers/linear->logistic 50)
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])
(defn get-ds
  []
  (mcore/ensure-images-on-disk!)
  (let [training-folder (str mcore/dataset-folder "training")
        test-folder (str mcore/dataset-folder "test")
        [train-ds test-ds] [(-> training-folder
                                (experiment-util/create-dataset-from-folder mcore/class-mapping)
                                (experiment-util/infinite-class-balanced-dataset))
                            (-> test-folder
                                (experiment-util/create-dataset-from-folder mcore/class-mapping))]]
    [train-ds test-ds]))

(defn train-forever
  ([inet filename ]
   (mcore/ensure-images-on-disk!)
   (println "Training forever.")
   (let [[train-ds test-ds] (get-ds)]
     (etrain/train-n
      inet 
      train-ds test-ds
      :epoch-count 3 
      :network-filestem filename))))

(defn save-interim-nets
  ([inet]
   (mcore/ensure-images-on-disk!)
   (let [[train-ds test-ds] (get-ds)
         train-fn #(etrain/train-n
                   % 
                   train-ds test-ds
                   :epoch-count  2  
                   :save-gradients? true)]
     (->> inet
          (iterate train-fn)
          (take 10)))))

(comment 
  (def res (mapv train-forever
                 [(initial-description-mnist 28 28 10)
                  (initial-description-mnist-wocl 28 28 10)]
                 ["mnist-with-cenloss" "mnist-wo-cenloss"]))
  (def res3 (save-interim-nets (lenet-mnist-wocl 28 28 10)))
  (def i2 (-> res3 count))
  (-> res3 second :cv-loss )
  (-> res3 )
  (def res4
    (let [i (save-interim-nets (-> res3 second))
          j (count i)]
      i))


  )
;;with cen-loss- best loss is 0.083
;;witout cen-los best loss is 0.100
;;(def res2 (mapv train-forever [(lenet-mnist-wocl 28 28 10)]))

(defn get-gradient-stats
  "returns a function applies on gradients at every layer"
  ([inp] (get-gradient-stats (juxt (partial apply max) (partial apply min)) inp))
  ([statfn inp]
   (dissoc (->> inp :traversal :buffers
                (mapv (fn[[k v]] {
                                  (-> k :id)
                                  (statfn
                                   (-> v :gradient m/to-vector))}))
                (apply merge))
           nil)))

(defn get-weight-stats
  ([inp](get-weight-stats (juxt (partial apply max) (partial apply min)) inp))
  ([statfn inp]
   (->> inp :compute-graph :buffers
        (mapv (fn[[k v]] {k (-> v :buffer seq m/to-vector statfn)}))
        (apply merge))))

(comment
  (mapv get-max-gradient res4)
  (->> res4 last :compute-graph :buffers (mapv (comp keys second )))

  (def relu1
    (let [i (save-interim-nets (mnist-relu 28 28 10))
          j (count i)]
      i))

  (def tanh1 (-> (save-interim-nets (mnist-tanh 28 28 10)) vec))

  ;;with weights initialized to -400 to 400
  (def tanh2 (-> (save-interim-nets (mnist-tanh 28 28 10)) vec))
  (def tanh3 (-> (save-interim-nets (mnist-tanh-woconv 28 28 10)) vec))

  )

(def init (-> (save-interim-nets (initial-description 28 28 10)) vec))

(get-gradient-stats (second init))

(def low-weight-fn #(->> % (mapv (fn[j] (Math/abs j)))
                         (filter (partial > 0.01))
                         count))

;(mapv get-max-weight res4)
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
                    shuffle
                    )
        network-bottleneck (network/dissoc-layers-from-network mnist-network :linear-2)
        layers-to-add (->
                       ;;[(layers/linear 2 :weight-initialization-type :constant)]
                        ;;[(layers/linear 2 :weight-initialization-type :xavier)]
                       [(layers/linear 2 :weight-initialization-type :constant)
                        (layers/linear 10)
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

;(def k1 (get-2d mnist-cenloss false))
;(def k1 (get-2d mnist-cenloss true))
;(->> k1 (mapv first ) set)
;(get-in (vec [(layers/linear 2)]) [0])
;(assoc-in (-> [(layers/linear 2)] flatten vec) [0 :parents] 10)
;(-> res :compute-graph :edges)
