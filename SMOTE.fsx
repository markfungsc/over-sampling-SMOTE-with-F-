#r "nuget: FSharp.Stats"
#load "dataType.fsx"
//https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node6.html
//https://github.com/Tommy-Zumstein/k-Nearest-Neighbor-Module/blob/master/knn.fs
open System
open FSharp.Stats.ML
open DataType

module SMOTE =
    [<CLIMutable>]
    type View = 
        {
            DTO: DateTimeOffset
            Return: float32
            MeanReturn: float32
            Std: float32
            BBANDWidth: float32
            Trima: float32
            MACD: float32
            MACDSignal: float32
            MACDHist: float32
            ADX: float32
            MFI: float32
            MOM: float32
            RSI: float32
            OBV: float32
            ATR: float32
            Label: uint32
        }

    let arrMean (arr: float[]) = 
        arr |> Array.average

    let arrMeanPlusStd (arr: float[]) (multiplier: float) = 
        let mean = arr |> Array.average
        let std = sqrt(Array.fold (fun acc elem -> acc + (pown (elem - mean) 2)) 0. arr / float arr.Length)
        mean + (std * multiplier)
    
    let smote (seed: int) (k: int) (nbOfSampleNeed: int) (minorityArr: float[][]) : float[][][] =
        let t = Array.length minorityArr //number of minority class
        let n = 
            if nbOfSampleNeed < 0 then failwithf "not minority"
            else
                int (Math.Ceiling (float nbOfSampleNeed / float t)) - 1 //amount of SMOTE
        let rnd = System.Random(Seed = seed)
    
        // KNN
        let findKNN (k: int) (i: float[]) (arr: float[][]) =
            [|0 .. (Array.length arr - 1)|]
            |> Array.Parallel.map (fun x -> arr.[x], DistanceMetrics.euclidean i arr.[x])
            |> Array.sortBy (fun (_, x) -> x)
            |> Array.take (k + 1)
            |> Array.tail
            |> Array.map (fun (x, _) -> x)

        let resultKNN =
            [|0 .. (t-1)|]
            |> Array.Parallel.map (fun x -> findKNN k minorityArr.[x] minorityArr)

        let genSyntheticArr (n: int) (arr: float[]) (knnArr: float[][]) : float[][] =            
            let rec genNewArr (n:int) (newArr: float[][]) (arr: float[]) (knnArr: float[][]) : float[][] = 
                match n with
                | 0 -> newArr
                | _ ->
                    let rndZeroToFour = rnd.Next(0, k)
                    let nn = knnArr.[rndZeroToFour]
                    let newPoint =
                        arr
                        |> Array.map2 (fun y x -> x + rnd.NextDouble() * (y - x)) nn
                    genNewArr (n-1) (Array.append newArr [|newPoint|]) arr knnArr

            genNewArr n [||] arr knnArr
                 
        minorityArr
        |> Array.Parallel.mapi (fun i x -> genSyntheticArr n x resultKNN.[i])

    let smoteForView (seed: int) (labels: uint) (k: int) (nbOfSampleNeed: int) (minorityView: View[]) : View[] =
        let minorityArr : float32[][] = 
            minorityView
            |> Array.Parallel.map (fun x -> [|x.Return; x.MeanReturn; x.Std; x.BBANDWidth; x.Trima; x.MACD; x.MACDSignal; x.MACDHist; x.ADX; x.MFI; x.MOM; x.RSI; x.OBV; x.ATR|])

        let t = Array.length minorityArr //number of minority class
        let n = 
            if nbOfSampleNeed < 0 then failwithf "not minority"
            else
                int (Math.Ceiling (float nbOfSampleNeed / float t)) - 1 //amount of SMOTE
        let rnd = System.Random(Seed = seed)
    
        // KNN
        let findKNN (k: int) (i: float32[]) (arr: float32[][]) =
            [|0 .. (Array.length arr - 1)|]
            |> Array.Parallel.map (fun x -> arr.[x], DistanceMetrics.euclidean i arr.[x])
            |> Array.sortBy (fun (_, x) -> x)
            |> Array.take (k + 1)
            |> Array.tail
            |> Array.map (fun (x, _) -> x)

        let resultKNN =
            [|0 .. (t-1)|]
            |> Array.Parallel.map (fun x -> findKNN k minorityArr.[x] minorityArr)

        let genSyntheticArr (n: int) (arr: float32[]) (knnArr: float32[][]) : float32[][] =            
            let rec genNewArr (n:int) (newArr: float32[][]) (arr: float32[]) (knnArr: float32[][]) : float32[][] = 
                match n with
                | 0 -> newArr
                | _ ->
                    let rndZeroToFour = rnd.Next(0, 5)
                    let nn = knnArr.[rndZeroToFour]
                    let newPoint =
                        arr
                        |> Array.map2 (fun y x -> x + (float32 (rnd.NextDouble())) * (y - x)) nn
                    genNewArr (n-1) (Array.append newArr [|newPoint|]) arr knnArr

            genNewArr n [||] arr knnArr
                 
        minorityArr
        |> Array.Parallel.mapi (fun i x -> genSyntheticArr n x resultKNN.[i])
        |> Array.concat
        |> Array.Parallel.map (
            fun x ->
                    {
                        DTO = DateTimeOffset.MinValue
                        Return = x.[0]
                        MeanReturn = x.[1]
                        Std = x.[2]
                        BBANDWidth = x.[3]
                        Trima = x.[4]
                        MACD = x.[5]
                        MACDSignal = x.[6]
                        MACDHist = x.[7]
                        ADX = x.[8]
                        MFI = x.[9]
                        MOM = x.[10]
                        RSI = x.[11]
                        OBV = x.[12]
                        ATR = x.[13]
                        Label = labels
                    }
                )

        

    

