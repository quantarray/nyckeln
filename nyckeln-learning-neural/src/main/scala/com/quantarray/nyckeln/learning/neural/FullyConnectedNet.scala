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

import scala.language.reflectiveCalls

/**
 * Fully-connected net.
 *
 * @author Araik Grigoryan
 */
case class FullyConnectedNet(activation: Activation, cost: Cost, connections: Seq[Synapse]) extends Net
{
  type C = Neuron

  type L = Nucleus

  type T = Synapse

  lazy val sourceLayerGroups: Map[Nucleus, Map[Neuron, Seq[Synapse]]] = connections.groupBy(_.source.layer).map((lc) => (lc._1, lc._2.groupBy(_.source)))

  lazy val targetLayerGroups: Map[Nucleus, Map[Neuron, Seq[Synapse]]] = connections.groupBy(_.target.layer).map((lc) => (lc._1, lc._2.groupBy(_.target)))

  /**
   * Creates a map of weights, in order of layer and target neuron index.
   */
  lazy val biases: Biases = props(targetLayerGroups, _.source.isBias, _.weight)

  /**
   * Creates a map of weights, in order of layer and source neuron index.
   */
  lazy val weights: Weights = props(sourceLayerGroups, _.source.nonBias, _.weight)
}

object FullyConnectedNet
{

  case class FromScratchBuilder(weight: WeightAssignment, activation: Activation, cost: Cost,
                                neuronsInLayer0: Int, neuronsInLayer1: Int, neuronsInLayer2AndUp: Int*) extends NetBuilder[Neuron, Synapse, FullyConnectedNet]
  {
    val layer0 = Nucleus(0, activation, neuronsInLayer0)

    val layer1 = Nucleus(1, activation, neuronsInLayer1)

    val layers2AndUp: Seq[Nucleus] = neuronsInLayer2AndUp.zipWithIndex.map(x => Nucleus(x._2 + 2, activation, x._1))

    val layers: Seq[Nucleus] = layer0 +: layer1 +: layers2AndUp

    val synapses: List[Synapse] = layers.zipWithIndex.foldLeft(List.empty[Synapse])((synapses, layerIndex) =>
    {
      if (layerIndex._1 == layers.last)
      {
        synapses
      }
      else
      {
        val sourceLayer = layerIndex._1
        val targetLayer = layers(layerIndex._2 + 1)

        val biasSynapses =
          targetLayer.cells.map(target => (Neuron(0, targetLayer, isBias = true), target)).map(st => Synapse(st._1, st._2, weight(st._1, st._2)))

        val weightSynapses = for
        {
          source <- sourceLayer.cells
          target <- targetLayer.cells
        } yield Synapse(source, target, weight(source, target))

        synapses ++ biasSynapses ++ weightSynapses
      }
    })

    override def net: FullyConnectedNet = FullyConnectedNet(activation, cost, synapses)
  }

  case class FromBiasesAndWeightsBuilder(activation: Activation, cost: Cost, biases: Biases, weights: Weights) extends NetBuilder[Neuron, Synapse, FullyConnectedNet]
  {
    val layers: Seq[Nucleus] = weights.map(kv => Nucleus(kv._1, activation, kv._2.size)).toSeq :+ Nucleus(weights.size, activation, biases.last._2.size)

    val synapses: List[Synapse] = layers.zipWithIndex.foldLeft(List.empty[Synapse])((synapses, layerIndex) =>
    {
      if (layerIndex._1 == layers.last)
      {
        synapses
      }
      else
      {
        val sourceLayer = layerIndex._1
        val targetLayer = layers(layerIndex._2 + 1)

        val biasSynapses =
          targetLayer.cells.map(target => (Neuron(0, targetLayer, isBias = true), target)).map(st =>
            Synapse(st._1, st._2, biases(targetLayer.index)(st._2.index).head))

        val weightSynapses = for
        {
          source <- sourceLayer.cells
          target <- targetLayer.cells
        } yield Synapse(source, target, weights(sourceLayer.index)(source.index)(target.index - 1))

        synapses ++ biasSynapses ++ weightSynapses
      }
    })

    override def net: FullyConnectedNet = FullyConnectedNet(activation, cost, synapses)
  }

  implicit val canBuildFrom = new NetCanBuildFrom[FullyConnectedNet, Neuron, Synapse, FullyConnectedNet]
  {
    /**
     * Creates a new builder on request of a net.
     */
    override def apply(from: FullyConnectedNet, biasesWeights: (Biases, Weights)): NetBuilder[Neuron, Synapse, FullyConnectedNet] =
    {
      FromBiasesAndWeightsBuilder(from.activation, from.cost, biasesWeights._1, biasesWeights._2)
    }

    /**
     * Creates a new builder from scratch.
     */
    override def apply(weightAssignment: WeightAssignment, activation: Activation, cost: Cost,
                       neuronsInLayer0: Int, neuronsInLayer1: Int, neuronsInLayer2AndUp: Int*): NetBuilder[Neuron, Synapse, FullyConnectedNet] =
    {
      FromScratchBuilder(weightAssignment, activation, cost, neuronsInLayer0, neuronsInLayer1, neuronsInLayer2AndUp: _*)
    }
  }

  /**
   * Connects layers in feed-forward fashion.
   *
   * In addition to the requested neurons, a bias cell will be created for each layer. By convention,
   * the zeroth layer will not receive a bias cell because it will directly absorb the inputs.
   */
  def apply(weightAssignment: WeightAssignment, activation: Activation, cost: Cost, neuronsInLayer0: Int, neuronsInLayer1: Int, neuronsInLayer2AndUp: Int*): FullyConnectedNet =
  {
    canBuildFrom(weightAssignment, activation, cost, neuronsInLayer0, neuronsInLayer1, neuronsInLayer2AndUp: _*).net
  }

  def apply(weightAssignment: WeightAssignment, regime: Regime, neuronsInLayer0: Int, neuronsInLayer1: Int, neuronsInLayer2AndUp: Int*): FullyConnectedNet =
  {
    apply(weightAssignment, regime.activation, regime.cost, neuronsInLayer0, neuronsInLayer1, neuronsInLayer2AndUp: _*)
  }
}
