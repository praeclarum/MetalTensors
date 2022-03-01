using System;
using System.Collections.Generic;
using System.Linq;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public class TrainingHistory
    {
        public BatchHistory[] Batches { get; }

        public TrainingHistory ()
        {
            Batches = Array.Empty<BatchHistory> ();
        }

        public TrainingHistory (IEnumerable<BatchHistory> batches)
        {
            Batches = batches.ToArray ();
        }

        /// <summary>
        /// The results of one batched execution of the model
        /// </summary>
        public class BatchHistory
        {
            public IMTLDevice Device { get; }
            public Tensor[] Results { get; }
            public Dictionary<string, float> Losses { get; }
            public Dictionary<string, Tensor[]> IntermediateValues { get; }

            public BatchHistory (Tensor[] results, Dictionary<string, float> losses, Dictionary<string, Tensor[]> intermediateValues, IMTLDevice device)
            {
                Results = results;
                Losses = losses;
                IntermediateValues = intermediateValues;
                Device = device;
            }

            public float AverageLoss {
                get {
                    var n = 0;
                    var sum = 0.0;
                    foreach (var r in Losses) {
                        sum += r.Value;
                        n += 1;
                    }
                    if (n > 0)
                        return (float)(sum / n);
                    return 0.0f;
                }
            }
        }
    }
}
