(ns cortex-examples.core
  (:require [loom.io :refer [dot-str view]]
            [cortex-examples.draw-graph :as dg]
            [loom.graph :as lg]
            [think.parallel.core :as parallel]
            [cortex.optimize :as opt]
            [cortex.optimize.adam :as adam]
            [cortex.nn.execute :as execute]
            [cortex.nn.compute-binding :as compute-binding]
            [cortex.nn.network :as network]
            [clojure.java.io :refer [file]]
            [clojure.java.shell :refer [sh]]
            [cortex.nn.traverse :as traverse]
            [cortex.graph :as graph]
            [cortex.nn.network :as network]
            [cortex.nn.layers :as layers]
            [cortex.util :as util]
            [cortex.loss.core :as loss]))

(def wdg (lg/weighted-digraph [:a :b 10] [:a :c 20] [:c :d 30] [:d :b 10]))
(dg/view (lg/digraph {1 [3 4] 2 [3 4] 3 [5 6] 4 [5 6]}) :map-layers {"one" [1 2]
                                                                     "two" [3 4]} )
(view (lg/digraph {1 [3 4] 2 [3 4] 3 [5 6] 4 [5 6]}) )
(view (lg/digraph {1 [3 4] 2 [3 4] 3 [5 6] 4 [5 6]}) )


(defn draw-nn
  [layer-cnt]
  (->> 
   (mapv #(mapv (fn[i] (str %2 i)) (range %1))
         layer-cnt 
         (iterate inc 1))
   (partition 2 1)
   (mapv (fn[[a b]]
           (apply merge (for [i a]
                          (assoc {} i b)))))
   (reduce merge)
   (lg/digraph)
   view
   ))

;(draw-nn [2 3 2])
;(draw-nn [5 10 2])




;(add-k layer-map)

;(render-to-bytes "aaaa" :fmt :a :alg :neato)
#_(let [dot (apply dot-str g (apply concat opts))
      {:keys [out]} (sh (name alg) (str "-T" (name fmt)) :in dot :out-enc :bytes)]
  out)
(def description
  [(layers/input 2 1 1 :id :data)
   (layers/batch-normalization)
   ;;Fix the weights to make the unit test work.
   (layers/linear 1 :weights [[-0.2 0.2]])
   (layers/logistic :id :labels)])


(def g (network/linear-network description))

(->> g :compute-graph :nodes
     #_(mapv (fn[[k v]] (select-keys v [:input-dimensions :output-dimensions]))))

(->> g :compute-graph :edges count)
(->> g :compute-graph :nodes count)
(->> g :compute-graph :nodes keys)

(->> g traverse/forward-traversal )
(->> g traverse/training-traversal :forward)

(defn mnist-initial-description
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

(def mnist (mnist-initial-description 28 28 10))

(def mnist-desc (network/linear-network mnist))

(def mnist-trav (traverse/training-traversal mnist-desc))
(def mnist-inf (traverse/inference-traversal mnist-desc))


(def lo (traverse/gradient-loss-function mnist-desc mnist-trav))
;;(def r1 (traverse/remove-non-trainable mnist-desc mnist-trav))
;;doesn't work

(-> (network/loss-function mnist-desc) )
(-> (network/print-layer-summary mnist mnist-trav) )
