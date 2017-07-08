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

import breeze.linalg.{sum, norm}
import breeze.numerics.log

/**
 * Cost function.
 *
 * @author Araik Grigoryan
 */
trait Cost extends ((Matrix, Matrix) => Double)
{
  /**
   * Partial derivative of the cost function with respect to output activation a.
   */
  def d(z: Matrix, a: Matrix, y: Matrix): Matrix
}

case class QuadraticCost(activation: Activation) extends Cost
{
  override def apply(a: Matrix, y: Matrix): Double =
  {
    val n = norm(a.toDenseVector - y.toDenseVector) // HACK: Breeze does not compute L2 norm of a DenseMatrix
    0.5 * n * n
  }

  override def d(z: Matrix, a: Matrix, y: Matrix): Matrix = (a - y) *:* activation.d(z)
}

case object CrossEntropyCost extends Cost
{
  override def apply(a: Matrix, y: Matrix): Double =
  {
    sum(y * log(a) - (Matrix.eye(y.rows) - y) * log(Matrix.eye(a.rows) - a))
  }

  override def d(z: Matrix, a: Matrix, y: Matrix): Matrix = a - y
}

case object LogLikelihoodCost extends Cost
{
  override def apply(a: Matrix, y: Matrix): Double =
  {
    sum(-log(a))
  }

  override def d(z: Matrix, a: Matrix, y: Matrix): Matrix = a - y
}

