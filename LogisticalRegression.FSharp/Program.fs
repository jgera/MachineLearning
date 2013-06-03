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

    // TODO:

    printfn "Done!"
    Console.ReadLine()

    0 // return an integer exit code
