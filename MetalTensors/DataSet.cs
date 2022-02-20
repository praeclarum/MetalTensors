using System;

using DataSetRow = System.ValueTuple<MetalTensors.Tensor[], MetalTensors.Tensor[]>;
using DataSetBatch = System.ValueTuple<MetalTensors.Tensor[][], MetalTensors.Tensor[][]>;
using System.Collections.Generic;
using Metal;

namespace MetalTensors
{
    public abstract class DataSet
    {
        public abstract int Count { get; }
        public abstract (Tensor[] Inputs, Tensor[] Outputs) GetRow (int index, IMTLDevice device);

        public static DataSet Generated (Func<int, IMTLDevice, DataSetRow> getRow, int count)
        {
            return new GeneratedDataSet (getRow, count);
        }

        public static DataSet Single (Tensor input, Tensor? output = null)
        {
            return new SingleDataSet (input, output);
        }

        public DataSet Subset (int index, int length)
        {
            return new SubsetDataSet (this, index, length);
        }

        public virtual DataSetBatch GetBatch (int index, int batchSize, IMTLDevice device)
        {
            var inputs = new List<Tensor[]> (batchSize);
            var outputs = new List<Tensor[]> (batchSize);
            var n = Count;
            for (var bi = 0; bi < batchSize; bi++) {
                var i = (index + bi) % n;
                var (ins, outs) = GetRow (i, device);
                inputs.Add (ins);
                outputs.Add (outs);
            }
            return (inputs.ToArray(), outputs.ToArray());
        }

        class GeneratedDataSet : DataSet
        {
            readonly int count;
            readonly Func<int, IMTLDevice, DataSetRow> getRow;

            public override int Count => count;
            public override DataSetRow GetRow (int index, IMTLDevice device) => getRow (index, device);

            public GeneratedDataSet (Func<int, IMTLDevice, DataSetRow> getRow, int count)
            {
                this.getRow = getRow;
                this.count = count;
            }
        }

        class SingleDataSet : DataSet
        {
            readonly DataSetRow row;

            public override int Count => 1;
            public override DataSetRow GetRow (int index, IMTLDevice device) => row;

            public SingleDataSet (Tensor input, Tensor? output)
            {
                var outputs = output != null ? new[] { output } : Array.Empty<Tensor> ();
                row = (new[] { input }, outputs);
            }
        }

        class SubsetDataSet : DataSet
        {
            readonly DataSet parent;
            readonly int index;
            readonly int length;

            public override int Count => length;
            public override DataSetRow GetRow (int index, IMTLDevice device) => parent.GetRow(this.index + index, device);

            public SubsetDataSet (DataSet parent, int index, int length)
            {
                this.parent = parent;
                this.index = index;
                this.length = length;
            }
        }
    }
}
