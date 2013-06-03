namespace LinearRegression
{
    using System;
    using System.IO;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;

    using Vector = MicrosoftResearch.Infer.Maths.Vector;

    class Program
    {
        static void Main()
        {
            Console.WriteLine("Loading data...");

            var data = (from line in File.ReadLines("ex1data2.txt")
                        let items = line.Split(',')
                        select items.Select(double.Parse).ToList()).ToList();

            var sourceX = DenseMatrix.Create(data.Count, data.First().Count - 1, (i, j) => data[i][j]);
            var y = DenseVector.Create(data.Count, i => data[i].Last());

            UseGradientDescent(sourceX, y);
            UseNormalEquation(sourceX, y);
            UseAdvancedOptimization(sourceX, y);

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void UseGradientDescent(DenseMatrix sourceX, DenseVector y)
        {
            Console.WriteLine("Normalizing features...");

            var normalizeResults = FeatureNormalize(sourceX);
            var xNormalized = normalizeResults.Item1;
            var mu = normalizeResults.Item2;
            var sigma = normalizeResults.Item3;
            var x = xNormalized.InsertColumn(0, DenseVector.Create(sourceX.RowCount, i => 1));

            Console.WriteLine("Running gradient descent...");

            var results = GradientDescent(x, y, DenseVector.Create(x.ColumnCount, i => 0), 0.01, 1000);
            var theta = results.Item1;

            Console.WriteLine("Theta computed (using gradient descent): {0}", theta);
            var price = new DenseVector(new[] { 1.0, (1650.0 - mu[0]) / sigma[0], (3.0 - mu[1]) / sigma[1] }) * theta;

            Console.WriteLine("Predicted price of a 1650 sq-ft, 3 br house (using gradient decent): {0}", price);
        }

        private static void UseNormalEquation(DenseMatrix sourceX, DenseVector y)
        {
            Console.WriteLine("Solving with normal equation...");

            var x = sourceX.InsertColumn(0, DenseVector.Create(sourceX.RowCount, i => 1));
            var theta = NormalEquation(x, y);

            Console.WriteLine("Theta computed (using normal equation): {0}", theta);
            var price = new DenseVector(new[] { 1.0, 1650.0, 3.0 }) * theta;

            Console.WriteLine("Predicted price of a 1650 sq-ft, 3 br house (using normal equation): {0}", price);
        }

        private static void UseAdvancedOptimization(DenseMatrix sourceX, DenseVector y)
        {
            Console.WriteLine("Normalizing features...");

            var normalizeResults = FeatureNormalize(sourceX);
            var xNormalized = normalizeResults.Item1;
            var mu = normalizeResults.Item2;
            var sigma = normalizeResults.Item3;
            var x = xNormalized.InsertColumn(0, DenseVector.Create(sourceX.RowCount, i => 1));

            Console.WriteLine("Running advanced optimization...");

            var theta = AdvancedOptimization(x, y, DenseVector.Create(x.ColumnCount, i => 0), 0);

            Console.WriteLine("Theta computed (using advanced optimization): {0}", theta);
            var price = new DenseVector(new[] { 1.0, (1650.0 - mu[0]) / sigma[0], (3.0 - mu[1]) / sigma[1] }) * theta;

            Console.WriteLine("Predicted price of a 1650 sq-ft, 3 br house (using advanced optimization): {0}", price);
        }

        private static Tuple<DenseMatrix, DenseVector, DenseVector> FeatureNormalize(Matrix<double> x)
        {
            var mu = DenseVector.Create(x.ColumnCount, i => x.Column(i).Mean());
            var sigma = DenseVector.Create(x.ColumnCount, i => x.Column(i).StandardDeviation());

            return Tuple.Create(
                DenseMatrix.Create(x.RowCount, x.ColumnCount, (i, j) => (x[i, j] - mu[j]) / sigma[j]), 
                mu, 
                sigma);
        }

        private static Vector<double> NormalEquation(Matrix<double> x, Vector<double> y)
        {
            var xTranspose = x.Transpose();
            return (xTranspose * x).Inverse() * xTranspose * y;
        }

        private static Tuple<Vector<double>, double[]> GradientDescent(Matrix<double> x, Vector<double> y, Vector<double> theta, double alpha, int numberOfIterations)
        {
            var m = y.Count;
            var jHistory = new double[numberOfIterations];
            
            for (int i = 0; i < numberOfIterations; i++)
            {
                theta = theta - (alpha / m) * x.Transpose() * (x * theta - y);
            }

            return Tuple.Create(theta, jHistory);
        }

        private static Vector<double> AdvancedOptimization(Matrix<double> x, Vector<double> y, Vector<double> theta, double lambda)
        {
            var m = x.RowCount;
            var featureCount = theta.Count;
            var xTranspose = x.Transpose();

            // TODO: convert all of this to use MicrosoftResearch.Infer.Maths instead of MathNet.Numerics
            var solver = new MicrosoftResearch.Infer.Maths.BFGS();
            var minTheta = solver.Run(
                MicrosoftResearch.Infer.Maths.DenseVector.FromList(theta),
                10000,
                (Vector vector, ref Vector dX) =>
                    {
                        var newTheta = DenseVector.Create(featureCount, n => vector[n]);

                        var regTheta = DenseVector.OfVector(newTheta);
                        regTheta[0] = 0;

                        var regThetaSq = DenseVector.OfVector(regTheta);
                        regThetaSq.MapInplace(t => t * t);

                        var h = x * newTheta - y;

                        var cost = ((h * h) / (2D * m)) + ((lambda / (2D * m)) * regThetaSq.Sum());

                        var grad = ((1D / m) * (xTranspose * h)) + ((lambda / m) * regTheta);

                        for (var j = 0; j < grad.Count; j++)
                        {
                            dX[j] = grad[j];
                        }

                        return cost;
                    });
  
            return DenseVector.Create(theta.Count, i => minTheta[i]);
        }
    }
}
