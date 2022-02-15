using System;

using DataSetRow = System.ValueTuple<MetalTensors.Tensor[], MetalTensors.Tensor[]>;

namespace MetalTensors
{
    public abstract class DataSet
    {
        public abstract int Count { get; }
        public abstract string[] Columns { get; }
        public abstract (Tensor[] Inputs, Tensor[] Outputs) GetRow (int index);

        public static DataSet Generated (Func<int, DataSetRow> getRow, int count, params string[] columns)
        {
            return new GeneratedDataSet (getRow, count, columns);
        }

        public static DataSet Single (string column, Tensor input, Tensor? output = null)
        {
            return new SingleDataSet (column, input, output);
        }

        public DataSet Subset (int index, int length)
        {
            return new SubsetDataSet (this, index, length);
        }

        class GeneratedDataSet : DataSet
        {
            readonly string[] columns;
            readonly int count;
            readonly Func<int, DataSetRow> getRow;

            public override int Count => count;
            public override string[] Columns => columns;
            public override DataSetRow GetRow (int index) => getRow (index);

            public GeneratedDataSet (Func<int, DataSetRow> getRow, int count, string[] columns)
            {
                this.getRow = getRow;
                this.columns = columns;
                this.count = count;
            }
        }

        class SingleDataSet : DataSet
        {
            readonly string[] columns;
            readonly DataSetRow row;

            public override int Count => 1;
            public override string[] Columns => columns;
            public override DataSetRow GetRow (int index) => row;

            public SingleDataSet (string column, Tensor input, Tensor? output)
            {
                columns = new[] { column };
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
            public override string[] Columns => parent.Columns;
            public override DataSetRow GetRow (int index) => parent.GetRow(this.index + index);

            public SubsetDataSet (DataSet parent, int index, int length)
            {
                this.parent = parent;
                this.index = index;
                this.length = length;
            }
        }
    }
}
