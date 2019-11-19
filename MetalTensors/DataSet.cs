using System;

namespace MetalTensors
{
    public abstract class DataSet
    {
        public abstract int Count { get; }
        public abstract string[] Columns { get; }
        public abstract Tensor[] GetRow (int index);

        public static DataSet Generated (Func<int, Tensor[]> getRow, int count, params string[] columns)
        {
            return new GeneratedDataSet (getRow, count, columns);
        }

        public static DataSet Single (string column, Tensor input)
        {
            return new SingleDataSet (column, input);
        }

        public DataSet Subset (int index, int length)
        {
            return new SubsetDataSet (this, index, length);
        }

        class GeneratedDataSet : DataSet
        {
            readonly string[] columns;
            readonly int count;
            readonly Func<int, Tensor[]> getRow;

            public override int Count => count;
            public override string[] Columns => columns;
            public override Tensor[] GetRow (int index) => getRow (index);

            public GeneratedDataSet (Func<int, Tensor[]> getRow, int count, string[] columns)
            {
                this.getRow = getRow;
                this.columns = columns;
                this.count = count;
            }
        }

        class SingleDataSet : DataSet
        {
            readonly string[] columns;
            readonly Tensor[] row;

            public override int Count => 1;
            public override string[] Columns => columns;
            public override Tensor[] GetRow (int index) => row;

            public SingleDataSet (string column, Tensor input)
            {
                columns = new[] { column };
                row = new[] { input };
            }
        }

        class SubsetDataSet : DataSet
        {
            readonly DataSet parent;
            readonly int index;
            readonly int length;

            public override int Count => length;
            public override string[] Columns => parent.Columns;
            public override Tensor[] GetRow (int index) => parent.GetRow(this.index + index);

            public SubsetDataSet (DataSet parent, int index, int length)
            {
                this.parent = parent;
                this.index = index;
                this.length = length;
            }
        }
    }
}
