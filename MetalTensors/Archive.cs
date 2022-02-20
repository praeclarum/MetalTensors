using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace MetalTensors
{
    public interface IHasBuffers
    {
        void ReadBuffers (ReadBuffer reader);
        void WriteBuffers (WriteBuffer writer);
    }

    public delegate void ReadBuffer (string name, Span<float> values);
    public delegate void WriteBuffer (string name, ReadOnlySpan<float> values);

    public class ArchiveReader
    {
        private readonly ZipArchive archive;

        public ArchiveReader (Stream stream)
        {
            archive = new ZipArchive (stream, ZipArchiveMode.Read, leaveOpen: true, Encoding.UTF8);
        }

        public T Read<T> () where T : Configurable
        {
            var entryName = $"{typeof (T).Name}.root.xml";
            var entry = archive.Entries.FirstOrDefault (x => x.FullName == entryName);
            if (entry == null)
                throw new Exception ($"Failed to find root entry");
            T value;
            var references = new Dictionary<int, Configurable> ();
            using (var s = entry.Open ()) {
                value = Config.Read<T> (s, references);
            }
            foreach (var r in references) {
                var c = r.Value;
                if (c is IHasBuffers hb) {
                    hb.ReadBuffers ((x, y) => ReadBuffer (c, x, y));
                }
            }
            return value;
        }

        void ReadBuffer (Configurable source, string bufferName, Span<float> values)
        {
            throw new NotImplementedException ($"Cannot read weights {source}.{bufferName}");
        }
    }

    public class ArchiveWriter : IDisposable
    {
        bool disposed = false;
        ZipArchive archive;

        public ArchiveWriter (Stream stream)
        {
            archive = new ZipArchive (stream, ZipArchiveMode.Create, leaveOpen: true, Encoding.UTF8);
        }

        public void Dispose ()
        {
            if (!disposed) {
                disposed = true;
                archive.Dispose ();
            }
        }

        public void Write (Configurable root)
        {
            if (disposed)
                throw new ObjectDisposedException (nameof (ArchiveWriter));
            var config = root.Config;
            var entryName = $"{config.ObjectType}.root.xml";
            var entry = archive.CreateEntry (entryName);
            var references = new HashSet<Configurable> ();
            using (var s = entry.Open ()) {
                config.Write (s, references);
            }
            foreach (var r in references) {
                if (r is IHasBuffers hb) {
                    var c = r;
                    hb.WriteBuffers ((x, y) => WriteBuffer (c, x, y));
                }
            }
        }

        void WriteBuffer (Configurable source, string bufferName, ReadOnlySpan<float> values)
        {
            throw new NotImplementedException ($"Cannot save weights {source}.{bufferName} = {values.Length}");
        }
    }
}
