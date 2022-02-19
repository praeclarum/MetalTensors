using System;

namespace MetalTensors
{
    /// <summary>
    /// <para>
    /// How to pad images when using convolutions.
    /// </para>
    /// <para>
    /// Let style = {-1,0,1} for valid, same, full;
    /// destinationSize = (sourceSize - 1) * stride + 1 + style * (filterWindowSize - 1);
    /// </para>
    /// <para>
    /// Most MPS neural network filters are considered forward filters.
    /// Some (for example, convolution transpose and unpooling) are considered reverse filters.
    /// For the reverse filters, the image stride is measured in destination values rather than source values
    /// and has the effect of enlarging the image rather than reducing it.
    /// When a reverse filter is used to "undo" the effects of a forward filter,
    /// the size policy should be the opposite of the forward padding method.
    /// For example, if the forward filter used Valid,
    /// the reverse filter should use Full.
    /// </para>
    /// </summary>
    public enum ConvPadding
    {
        Valid = -1,
        Same = 0,
        Full = 1,
    }
}
