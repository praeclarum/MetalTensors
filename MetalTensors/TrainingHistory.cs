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
            public MPSImage[] SourceImages { get; private set; }

            public BatchHistory (Tensor[] results, Dictionary<string, float> losses, Dictionary<string, Tensor[]> intermediateValues, MPSImage[] sourceImages, IMTLDevice device)
            {
                Results = results;
                Losses = losses;
                IntermediateValues = intermediateValues;
                SourceImages = sourceImages;
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

            /// <summary>
            /// This is a convenience method to dispose of any temporary Metal images (SourceImages)
            /// created during the execution of the model. It's optional to call this
            /// (and don't call it if you need to reuse the inputs to the network)
            /// but can help in memory constrained environments.
            /// </summary>
            public void DisposeSourceImages ()
            {
                var old = SourceImages;
                SourceImages = Array.Empty<MPSImage> ();
                if (old != null) {
                    for (var i = 0; i < old.Length; i++) {
                        old[i]?.Dispose ();
                    }
                }
            }
        }
    }
}
