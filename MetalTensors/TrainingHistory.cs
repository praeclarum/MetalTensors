using System;
using System.Collections.Generic;
using System.Linq;

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

        public class BatchHistory
        {
            public Tensor[] Results { get; }
            public Tensor[] Loss { get; }
            public Dictionary<string, Tensor[]> IntermediateValues { get; }

            public BatchHistory (Tensor[] results, Tensor[] loss, Dictionary<string, Tensor[]> intermediateValues)
            {
                Results = results;
                Loss = loss;
                IntermediateValues = intermediateValues;
            }
        }
    }
}
