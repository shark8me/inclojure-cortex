---
layout: default
title: An introduction to Cortex
---

# An introduction to Cortex

This is the source repository for a presentation at inClojure 2018 at Bangalore, on building Neural Networks in Clojure using the [Cortex](https://github.com/thinktopic/cortex) library.

* Slides are [available online](https://shark8me.github.io/inclojure-cortex/cortex.html)
* The [org-mode](https://github.com/shark8me/inclojure-cortex/blob/master/docs/cortex.org) source.


## Exploring Cortex: a series of articles

The following notebooks document my REPL-explorations on Cortex

$ cd cortex-examples

$ lein gorilla :port 37128 :nrepl-port 20345

(or choose your own ports)

As an example, the vanishing sigmoid worksheet can be opened at this local link in your browser http://127.0.0.1:37128/worksheet.html?filename=ws/vanishing_sigmoid.cljw 

## Notebooks:

* [Building a network](http://viewer.gorilla-repl.org/view.html?source=github&user=shark8me&repo=inclojure-cortex&path=cortex-examples/ws/occupancy.cljw) to train the room occupancy dataset 
* [Comparing different activation](http://viewer.gorilla-repl.org/view.html?source=github&user=shark8me&repo=inclojure-cortex&path=cortex-examples/ws/matrix_mult.cljw) functions in Cortex
* [Replicating](http://viewer.gorilla-repl.org/view.html?source=github&user=shark8me&repo=inclojure-cortex&path=cortex-examples/ws/vanishing_sigmoid.cljw) the [vanishing sigmoid problem](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b). We explore the gradients and weights of the inner layers .


## License

Copyright © 2018 shark8me 

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
