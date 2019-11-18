using System;
namespace MetalTensors.Layers
{
    public abstract class BinopLayer : Layer
    {
        public override int MinInputCount => 2;

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            var ashape = inputs[0].Shape;
            var bshape = inputs[1].Shape;

            if (ashape.Length != bshape.Length)
                throw new ArgumentException ($"Binary operands must have matching shape dimensions.  {ashape.Length} and {bshape.Length} provided", nameof (inputs));

            for (var i = 0; i < ashape.Length; i++) {
                if (ashape[i] != bshape[i]) {
                    throw new ArgumentException ($"Binary operands must have matching shapes. {ashape.ToShapeString ()} and {bshape.ToShapeString ()} provided", nameof (inputs));
                }
            }
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }
    }
}
