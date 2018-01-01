(defproject cortex-examples "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins  [[lein-gorilla "0.4.0"]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/experiment "0.9.23-SNAPSHOT"]
                 [mnist-classification "0.9.23-SNAPSHOT"]
                 [org.clojure/tools.cli "0.3.5"]
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 [org.shark8me/tfevent-sink "0.1.4"]
                 [cc.artifice/clj-ml "0.8.5"]
                 [com.rpl/specter "1.0.5"]
                 [org.clojure/data.csv "0.1.3"]
                 [core-matrix-gorilla "0.1.0"]
                 [net.mikera/core.matrix "0.61.0"]  
                 ;viz
                 [aysylu/loom "1.0.0"]
                 ;;metrics
                 [datacraft-sciences/confuse "0.1.1"]])
