using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CommonLib;

namespace TestNeuralNetwork
{
    public static class NeuralNetwork
    {
        public static Matrix Sigmoid(Matrix Z)
        {
            return 1 / ((Math.Exp(1) ^ (Z * (-1))) + 1);
        }

        public static Matrix SigmoidGradient(Matrix Z)
        {
            return Sigmoid(Z).MultiplyElementWise((1 - Sigmoid(Z)));
        }

        public static Matrix HypothesisFunction(Matrix X, Matrix theta)
        {
            var Z = X * theta;
            return Sigmoid(Z);
        }

        public static Tuple<double, Matrix> CostFunction(Matrix theta, Matrix X, Matrix y, double lambda = 1)
        {
            var h = HypothesisFunction(X, theta);

            double m = (double)y.Rows;

            var y1 = (~y) * Matrix.Log(h);

            var y0 = (~(1 - y)) * Matrix.Log(1 - h);

            //J = (-1/m) * (sum(y' * log(h) + (1 - y)' * log(1-h))) + (lambda/(2*m)) * sum(theta(2:size(theta)).^2)';

            var theta0 = theta.SubMatrix(1, theta.Rows, false);

            var J = Matrix.Sum(y1 + y0) * (double)(-1 / m) + Matrix.Sum(theta0 ^ 2) * (double)(lambda / (2 * m));

            // x_o = X(:,1);
            // x_rest = X(:, 2:end);

            // grad(1) = (1/m) * sum((h - y)'*x_o, 1);
            // grad(2:end) = (1/m) * sum(h - y)'*x_rest, 1)' + (lambda/m) * theta(2:end);

            var x0 = X.SubMatrix(0, 1);

            var xRest = X.SubMatrix(1, X.Columns);

            var grad1 = (double)(1 / m) * Matrix.Sum((~(h - y)) * x0);
            var grad2 = (double)(1 / m) * Matrix.Sum((~(h - y)) * xRest) + (double)(lambda / m) * (~theta0);

            var grad = Matrix.ConcatMatrices(grad1, grad2);

            var result = Tuple.Create(J[0, 0], grad);

            return result;
        }

        // Gradient Descent
        public static Tuple<Matrix, Matrix> GradientDescent(CostFunctionWithThetaParameter func, Matrix theta, double alpha, int numberIterations)
        {
            var JHistory = new Matrix(numberIterations, 1);
            var alphaHistory = new double[numberIterations];
            for (int i = 0; i < numberIterations; i++)
            {
                var res = func(theta);

                var h = res.Item1;
                var grad = res.Item2;
                JHistory[i, 0] = h;

                //double[] temp = new double[theta.Rows];
                //for (int j = 0; j < theta.Rows; j++)
                //{
                //    temp[j] = theta[j, 0] - (grad[j, 0] * alpha);
                //}

                //theta = new Matrix(theta.Rows, 1);
                //for (int j = 0; j < theta.Rows; j++)
                //{
                //    theta[j, 0] = temp[j];
                //}

                theta = theta - grad * alpha;

                // "bold driver" - if we decrease the cost function, increase the learning rate by 5% but
                // in case when we increase the cost function, decrease the learning rate by 50%
                if (i > 0)
                {
                    if (JHistory[i, 0] < JHistory[i - 1, 0])
                    {
                        alpha += 0.05 * alpha;
                    }
                    else
                    {
                        alpha -= 0.5 * alpha;
                    }
                }

                alphaHistory[i] = alpha;

                if (i > 0 && JHistory[i, 0] < JHistory[i - 1, 0] &&
                    Equalities.DoubleEquals(JHistory[i, 0], JHistory[i - 1, 0]))
                {
                    break;
                }   
            }

            var m = new Matrix(numberIterations, 1, alphaHistory);
            return Tuple.Create(theta, JHistory);
        }

        public delegate Tuple<double, Matrix> CostFunctionWithThetaParameter(Matrix m);

        public static Matrix ComputeNumericalGradient(CostFunctionWithThetaParameter J, Matrix theta)
        {
            double epsilon = 0.0001;

            var numericalGradient = new Matrix(theta.Rows, 1);
            var perturbations = new Matrix(theta.Rows, 1); // смущения ;))

            for (int p = 0; p < theta.Rows; p++)
            {
                perturbations[p, 0] = epsilon;

                double loss1 = J(theta + perturbations).Item1;
                double loss2 = J(theta - perturbations).Item1;

                numericalGradient[p, 0] = ((loss1 - loss2) / (double)(2 * epsilon));

                perturbations[p, 0] = 0;
            }

            return numericalGradient;
        }

        // Forward Propagation
        public static List<Matrix> ForwardPropagation(Matrix X, IEnumerable<Matrix> thetas)
        {
            var activationFunctions = new List<Matrix>();

            var ai = X;
            for (int i = 0; i < thetas.Count(); i++)
            {
                var thetai = thetas.ElementAt(i);

                var onesT = new Matrix(1, 1, thetai.Columns);
                thetai = Matrix.ConcatMatrices(onesT, thetai, false);

                var ones = new Matrix(1, ai.Rows, 1);
                ai = Matrix.ConcatMatrices(ones, ai);

                ai = HypothesisFunction(ai, thetai);

                activationFunctions.Add(ai);
            }

            return activationFunctions;
        }

        public static Matrix GetOutputVectors(Matrix yClasses, int numberLabels)
        {
            int m = yClasses.Rows;
            
            var y = new Matrix(m, numberLabels);

            for(int i=0; i<m; i++)
            {
                y[i, (int)yClasses[i, 0]-1] = 1;
            }

            return y;
        }

        public static Tuple<double, Matrix> BackPropagation(Matrix theta, int inputLayerSize, int hiddenLayerSize, int numberLabels,
            Matrix X, Matrix y, double lambda)
        {
            int m = X.Rows;

            var theta1 = theta.SubMatrix(0, hiddenLayerSize * (inputLayerSize + 1), false);
            theta1.Reshape(hiddenLayerSize, inputLayerSize + 1);

            var theta2 = theta.SubMatrix(hiddenLayerSize * (inputLayerSize + 1), theta.Rows, false);
            theta2.Reshape(numberLabels, hiddenLayerSize + 1);
         
            /// forward propagation   
            var a1 = X;
            var onesa1 = new Matrix(1, a1.Rows, 1);
            a1 = Matrix.ConcatMatrices(onesa1, a1);

            var z2 = a1 * ~theta1;
            var a2 = Sigmoid(z2);

            var onesa2 = new Matrix(1, a2.Rows, 1);
            a2 = Matrix.ConcatMatrices(onesa2, a2);

            var z3 = a2 * ~theta2;
            var a3 = Sigmoid(z3);

            var h = a3;
            /// end of forward propagation

            var newY = GetOutputVectors(y, h.Columns);

            double J = BackPropagationCostFunction(h, newY,
                inputLayerSize, hiddenLayerSize, numberLabels,
                theta1, theta2, lambda, m);
         
            /// back propagation
            var delta3 = h - newY;
            var r2 = ~theta2.SubMatrix(1, theta2.Columns) * ~delta3;

            var delta2 = (~r2).MultiplyElementWise(SigmoidGradient(z2));

            var theta1_grad = ~delta2 * a1;
            var theta2_grad = ~delta3 * a2;
 
            var t1 = (double) lambda * theta1 /  m;
            t1.Zero(0, false);

            var t2 = (double) lambda * theta2 / m;
            t2.Zero(0, false);

            theta1_grad = theta1_grad / m + t1;
            theta2_grad = theta2_grad / m + t2;
            
            var theta1_grad_arr = theta1_grad.GetSingleArrayAsColumns();
            var theta2_grad_arr = theta2_grad.GetSingleArrayAsColumns();

            var grad = theta1_grad_arr.Concat(theta2_grad_arr).ToArray();
            var gradMatrix = new Matrix(grad.Count(), 1, grad);

            return Tuple.Create(J, gradMatrix);
        }

        private static double BackPropagationCostFunction(Matrix h, Matrix y,
            int inputLayerSize, int hiddenLayerSize, int numberLabels, 
            Matrix theta1, Matrix theta2, double lambda, int numberOfTestSamples)
        {
            int m = numberOfTestSamples;
         
            double cost = 0;
            for (int i = 0; i < m; i++)
            {
                for (int l = 0; l < numberLabels; l++)
                {
                    var partA = y[i, l] * Math.Log(h[i, l]);
                    var partB = (1 - y[i, l]) * Math.Log(1 - h[i, l]);
                    cost = cost + partA + partB;
                }
            }
            double J = cost * (double)(-1 / (double)m);

            double toAdd = GetRegularisationTerm(inputLayerSize, hiddenLayerSize, numberLabels,
                theta1, theta2, lambda, m);
            
            J = J + toAdd;

            return J;
        }

        private static double GetRegularisationTerm(int inputLayerSize, int hiddenLayerSize, int numberLabels,
            Matrix theta1, Matrix theta2, double lambda, double numberTestSamples)
        {
            var m = numberTestSamples;

            //var zeros1 = new Matrix(theta1.Rows, 1);
            //zeros1.Zero();
            //var zeros2 = new Matrix(theta2.Rows, 1);
            //zeros2.Zero();

            //var thetaNew1 = Matrix.ConcatMatrices(zeros1, theta1.SubMatrix(1, theta1.Columns));
            //var thetaNew2 = Matrix.ConcatMatrices(zeros2, theta2.SubMatrix(1, theta2.Columns));

            double thetaSumA = 0;
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                for (int k = 1; k < inputLayerSize + 1; k++) // without first column - bias term!
                {
                    thetaSumA += Math.Pow(theta1[j, k], 2);
                }
            }

            double thetaSumB = 0;
            for (int j = 0; j < numberLabels; j++)
            {
                for (int k = 1; k < hiddenLayerSize + 1; k++) // without first column - bias term!
                {
                    thetaSumB += Math.Pow(theta2[j, k], 2);
                }
            }

            var toAdd = ((double)(lambda / ((double)(2 * m)))) * (thetaSumA + thetaSumB);

            return toAdd;
        }
    }
}
