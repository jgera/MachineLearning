namespace LogisticalRegression
{
    using System;
    using System.IO;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Loading data...");

            var data = (from line in File.ReadLines("ex2data1.txt")
                        let items = line.Split(',')
                        select items.Select(double.Parse).ToList()).ToList();

            var sourceX = DenseMatrix.Create(data.Count, data.First().Count - 1, (i, j) => data[i][j]);
            var y = DenseVector.Create(data.Count, i => data[i].Last());

            // TODO:

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static double Sigmoid(double value)
        {
            return 1.0 / (1.0 - Math.Exp(-value));
        }

        private static Tuple<double, DenseVector> RegularizedLogisticalRegressionCostFunction(
            DenseVector theta, DenseMatrix X, DenseVector y, double lamda)
        {
            int m = y.Count;
            double J = 0.0;
            DenseVector grad = DenseVector.Create(theta.Count, i => 0.0);

            // TODO:

            return Tuple.Create(J, grad);
        }

        private DenseVector Predict(Vector<double> theta, Matrix<double> X)
        {
            var temp = (theta * X.Transpose());
            
            throw new NotImplementedException();
        }
    }
}