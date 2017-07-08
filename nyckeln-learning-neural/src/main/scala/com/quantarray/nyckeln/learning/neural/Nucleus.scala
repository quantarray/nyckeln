/*
 * Nyckeln
 * http://nyckeln.io
 *
 * Copyright 2012-2017 Quantarray, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.quantarray.nyckeln.learning.neural

import scala.collection.immutable

/**
 * Nucleus. A compact collection of neurons forming a layer inside a net.
 *
 * @author Araik Grigoryan
 */
case class Nucleus(index: LayerIndex, activation: Activation, numberOfNeurons: Int) extends Layer
{
  type C = Neuron

  lazy val cells: immutable.IndexedSeq[Neuron] = (1 to numberOfNeurons).map(Neuron(_, this))

  override def toString: String = s"layer $index"
}
