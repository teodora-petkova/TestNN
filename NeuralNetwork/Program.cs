using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CommonLib;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Data.Text;
using System.IO;

namespace TestNeuralNetwork
{
    public class Program
    {
        static void Main(string[] args)
        {
            Utils.SetNumberDecimalSeparatorToDotInCultureSettings();

            var x_data = DelimitedReader.Read<double>(Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\x_data.txt"));
            var y_data = DelimitedReader.Read<double>(Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\y_data.txt"));

            NeuralNetwork.CostFunctionWithThetaParameter backProp =
                tt =>
                {
                    var backPropagationResult = NeuralNetwork.BackPropagation(x_data, y_data, tt, new List<int>() { 25 }, 10, 0);
                    return backPropagationResult;
                };

            var thetas = NeuralNetwork.RandomInitialiseWeights(400, 10, new List<int>() { 25 });

            var theta = NeuralNetwork.PackThetas(thetas);

            var resultGradientDescent = NeuralNetwork.GradientDescent(backProp, theta, 1, 300);
            DelimitedWriter.Write("..\\..\\TestData\\ThetaGradient.txt", resultGradientDescent.Item1, "thetaGradient");
            DelimitedWriter.Write("..\\..\\TestData\\JHistory.txt", resultGradientDescent.Item2, "JHistory");

            var resultTheta = resultGradientDescent.Item1;

            //var resultComputeNumericalGradient = NeuralNetwork.ComputeNumericalGradient(backProp, resultTheta);
            //resultComputeNumericalGradient.Save("D:/test/numericalGradients.txt");

            var result5 = NeuralNetwork.BackPropagation(x_data, y_data, resultTheta, new List<int>(25), 10, 0);
            var JJ = result5.Item1;
            DelimitedWriter.Write("..\\..\\TestData\\GradientForBackPropagation.txt", result5.Item2, "gradBackPropagation");

            var resultThetaLoadedFromFile = (DenseMatrix)DelimitedReader.Read<double>("..\\..\\TestData\\thetaGradient.txt");
            var thetasList = NeuralNetwork.UnpackThetas(resultThetaLoadedFromFile, 400, new List<int>() { 25 }, 10);

            var result = NeuralNetwork.GetPredictions(x_data, y_data, thetasList); 

            Console.WriteLine(result.Item1);

            DelimitedWriter.Write("..\\..\\TestData\\Predictions.txt", result.Item2, "");
        }
    }
}
