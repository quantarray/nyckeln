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

package com.quantarray.nyckeln.learning

import breeze.linalg.support.LiteralRow
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero

import scala.collection.immutable.SortedMap
import scala.reflect.ClassTag

package object neural
{
  type CellIndex = Int

  type LayerIndex = Int

  // Map of Cell's index to a Seq of properties
  type LayerPropMap[T] = Map[CellIndex, Seq[T]]

  object LayerPropMap
  {
    def empty[T]: SortedMap[CellIndex, Seq[T]] = SortedMap.empty[CellIndex, Seq[T]]
  }

  // Map of Layer's index to a LayerMap
  type NetPropMap[T] = Map[LayerIndex, LayerPropMap[T]]

  object NetPropMap
  {
    def empty[T]: SortedMap[LayerIndex, LayerPropMap[T]] = SortedMap.empty[LayerIndex, LayerPropMap[T]]
  }

  type Biases = NetPropMap[Double]
  val Biases = NetPropMap

  type Weights = NetPropMap[Double]
  val Weights = NetPropMap

  // Fitness function: determines if output Seq[Double] is fit with respect to expected sample
  type Fitness = (Seq[Double], SupervisedDataSample) => Boolean

  /**
   * Breeze types
   */

  type Vector = DenseVector[Double]

  object Vector
  {
    def apply[V: ClassTag](values: Seq[V]) = DenseVector(values: _*)
  }

  type Matrix = DenseMatrix[Double]

  object Matrix
  {
    def apply[@specialized R, @specialized(Double, Int, Float, Long) V](rows: Seq[R])(implicit rl: LiteralRow[R, V], man: ClassTag[V], zero: Zero[V]): DenseMatrix[V] = DenseMatrix[R, V](rows: _*)

    def zeros(rows: Int, cols: Int): DenseMatrix[Double] = DenseMatrix.zeros[Double](rows, cols)

    def eye(n: Int): DenseMatrix[Double] = DenseMatrix.eye[Double](n)
  }

}
