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

import com.quantarray.nyckeln.learning.{SupervisedDataSample, SupervisedDataSet}
import org.scalatest.{FlatSpec, Matchers}

/**
 * Fully-connected net spec.
 *
 * @author Araik Grigoryan
 */
class FullyConnectedNetSpec extends FlatSpec with Matchers
{
  "Fully-connected net" should "be trainable with 3 layers" in
    {
      val net = FullyConnectedNet(IndexedWeightAssignment, SigmoidActivation, QuadraticCost(SigmoidActivation), 4, 3, 2)

      net.connections.size should be((4 + 1) * 3 + (3 + 1) * 2) // +1s are to account for the Biases

      val weights = net.weights
      weights.size should be(2)
      weights(0).size should be(4)
      weights(1).size should be(3)

      val biases = net.biases
      biases.size should be(2)
      biases(1).size should be(3)
      biases(2).size should be(2)

      val trainingSet = new SupervisedDataSet
      {
        override def samples: Seq[SupervisedDataSample] =
          Seq.fill(10000)(
            new SupervisedDataSample
            {
              override def input: Seq[Double] = Seq(1, 2, 3, 4)

              override def target: Seq[Double] = Seq(4, 2.5)
            }
          )
      }

      val trainer = BackPropagationTrainer(0.3, 0.5)

      val trainedNets = trainer.trainAndTest(net, 1, 1, trainingSet)

      trainedNets.size should equal(2)
      trainedNets.last._1.connections.size should equal(net.connections.size)
    }

  it should "be trainable with 4 layers" in
    {
      val net = FullyConnectedNet(IndexedWeightAssignment, SigmoidActivation, QuadraticCost(SigmoidActivation), 5, 4, 3, 2)

      net.connections.size should be((5 + 1) * 4 + (4 + 1) * 3 + (3 + 1) * 2) // +1s are to account for the Biases

      val weights = net.weights
      weights.size should be(3)
      weights(0).size should be(5)
      weights(1).size should be(4)
      weights(2).size should be(3)

      val biases = net.biases
      biases.size should be(3)
      biases(1).size should be(4)
      biases(2).size should be(3)
      biases(3).size should be(2)

      val trainingSet = new SupervisedDataSet
      {
        override def samples: Seq[SupervisedDataSample] =
          Seq(
            new SupervisedDataSample
            {
              override def input: Seq[Double] = Seq(1, 2, 3, 4, 5)

              override def target: Seq[Double] = Seq(4, 2.5)
            }
          )
      }

      val trainer = BackPropagationTrainer(0.3, 0.5)

      val trainedNets = trainer.trainAndTest(net, 1, 1, trainingSet)

      trainedNets.size should equal(2)
      trainedNets.last._1.connections.size should equal(net.connections.size)
    }
}
