using System.Collections.Generic;
using System.Linq;

namespace MetalTensors
{
    public class TrainingHistory
    {
        public BatchHistory[] Batches { get; }

        public TrainingHistory (IEnumerable<BatchHistory> batches)
        {
            Batches = batches.ToArray ();
        }

        public class BatchHistory
        {
            public Tensor[] Loss { get; }
            public Dictionary<string, Tensor[]> IntermediateValues { get; }

            public BatchHistory (Tensor[] loss, Dictionary<string, Tensor[]> intermediateValues)
            {
                Loss = loss;
                IntermediateValues = intermediateValues;
            }
        }
    }
}
