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

        class GeneratedDataSet : DataSet
        {
            private readonly string[] columns;
            private readonly int count;
            private readonly Func<int, Tensor[]> getRow;

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
            private string[] columns;
            private Tensor[] row;

            public override int Count => 1;
            public override string[] Columns => columns;
            public override Tensor[] GetRow (int index) => row;

            public SingleDataSet (string column, Tensor input)
            {
                columns = new[] { column };
                row = new[] { input };
            }
        }
    }
}
