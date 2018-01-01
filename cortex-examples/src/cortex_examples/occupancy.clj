(ns cortex-examples.occupancy
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


(defn load-data-vectors
  []
  (->> "resources/occupancy/datatraining.csv"
                       (slurp)
                       (clojure.string/split-lines)
                       (rest)                                     ;; ignore the header row
                       (map (fn [l] (drop 2 (clojure.string/split l #"," ))))  ;; id, date 
                       (mapv (fn [m] (mapv #(Double. %) m)))))

(defn make-feature-vec
  [data-vectors]
  (->> data-vectors 
       (mapv (fn[m] {:data (-> m butlast vec) :labels (-> m last vector)}))
       shuffle))

(defn train-test-ds
  []
  (let [ds (make-feature-vec (load-data-vectors))
        ds-count (count ds)
        ;;do a 90-10 split into train/test
        cutoff (int (* 0.9 ds-count))
        train-ds (take cutoff ds)
        test-ds (drop cutoff ds)]
    [train-ds test-ds]))

(def description
  [(layers/input 5 1 1 :id :data)
   (layers/batch-normalization)
   (layers/linear 1)
   (layers/logistic :id :labels)])


