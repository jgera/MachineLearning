open System
open System.IO
open System.Linq
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Generic.FSharpExtensions
open MathNet.Numerics.Statistics

[<EntryPoint>]
let main argv = 
    printfn "Loading data..."

    let data = File.ReadLines "ex1data2.txt" 
               |> Seq.map (fun s -> s.Split ',')
               |> Array.ofSeq
               |> Array.map (fun a -> Array.map Double.Parse a)
    
    // There has got to be a more elegant way to populate these from a CSV file in F#...
    let sourceX = DenseMatrix.init data.Length (data.First().Length - 1) (fun i j -> data.[i].[j])
    let y = DenseVector.init data.Length (fun i -> data.[i].Last())

    let normalEquation (x : Matrix<float>) (y : Vector<float>) =
        let xTranspose = x.Transpose()
        (xTranspose * x).Inverse() * xTranspose * y

    let useNormalEquation (sourceX : DenseMatrix) (y : DenseVector) =
        printfn "Solving with normal equation..."

        let x = sourceX.InsertColumn(0, (DenseVector.create sourceX.RowCount 1.0))
        let theta = normalEquation x y

        printfn "Theta computed (using normal equation): %A" theta
        let price = (DenseVector.raw [|1.0; 1650.0; 3.0;|]) * theta

        printfn "Predicted price of a 1650 sq-ft, 3 br house (using normal equation): %f" price

    let featureNormalize (x : Matrix<float>) =
        let mu = DenseVector.init x.ColumnCount (fun i -> x.Column(i).Mean())
        let sigma = DenseVector.init x.ColumnCount (fun i -> x.Column(i).StandardDeviation())
        let xNormalized = DenseMatrix.init x.RowCount x.ColumnCount (fun i j -> (x.[i,j] - mu.[j]) / sigma.[j])
        (xNormalized, mu, sigma)

    let gradientDescent (x : Matrix<float>) (y : Vector<float>) (theta: Vector<float>) (alpha : float) (numberOfIterations : int) =
        let m = float y.Count
        let jHistory = Array.zeroCreate numberOfIterations
        let mutable mTheta = theta

        for i in [0..numberOfIterations] do
            mTheta <- mTheta - (alpha / m) * x.Transpose() * (x * mTheta - y)
            //jHistory.[i] <- (x * mTheta - y)' * (x * theta - y) / (2.0 * m)

        (mTheta, jHistory)

    let useGradientDescent (sourceX : DenseMatrix) (y : DenseVector) =
        printfn "Normalizing features..."

        let (xNormalized, mu, sigma) = featureNormalize sourceX
        let x = xNormalized.InsertColumn(0, (DenseVector.create sourceX.RowCount 1.0))

        printfn "Running gradient desecent...."

        let (theta, jHistory) = gradientDescent x y (DenseVector.zeroCreate x.ColumnCount) 0.01 1000

        printfn "Theta computed (using gradient descent): %A" theta
        let price = DenseVector.raw [|1.0; ((1650.0 - mu.[0]) / sigma.[0]); ((3.0 - mu.[1]) / sigma.[1])|] * theta
        
        printfn "Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): %f" price

    useGradientDescent sourceX y
    useNormalEquation sourceX y

    printfn "Done!"
    Console.ReadLine()

    0 // return an integer exit code
