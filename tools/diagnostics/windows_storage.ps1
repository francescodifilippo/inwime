param(
    [ValidateSet('usb','read','counters')][string]$Mode='counters',
    [string]$Path,
    [int]$Repeats=10,
    [int]$BufferMiB=64
)
$ErrorActionPreference='Stop'
if ($Mode -eq 'counters') {
    Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk |
        Select-Object Name,DiskReadBytesPersec,DiskWriteBytesPersec | ConvertTo-Json -Compress
    exit
}
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class InWiMeStorage {
    [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)]
    static extern SafeFileHandle CreateFile(string path,uint access,uint share,IntPtr security,uint disposition,uint flags,IntPtr template);
    [DllImport("kernel32.dll",SetLastError=true)]
    static extern IntPtr VirtualAlloc(IntPtr address,UIntPtr size,uint kind,uint protect);
    [DllImport("kernel32.dll",SetLastError=true)]
    static extern bool VirtualFree(IntPtr address,UIntPtr size,uint kind);
    [DllImport("kernel32.dll",SetLastError=true)]
    static extern bool ReadFile(SafeFileHandle file,IntPtr buffer,uint count,out uint read,IntPtr overlapped);
    [DllImport("kernel32.dll",SetLastError=true)]
    static extern bool GetFileSizeEx(SafeFileHandle file,out long size);
    [DllImport("kernel32.dll",SetLastError=true)]
    static extern bool DeviceIoControl(SafeFileHandle file,uint code,byte[] input,uint inputSize,byte[] output,uint outputSize,out uint returned,IntPtr overlapped);
    public static double Read(string path,int bufferMiB) {
        if(bufferMiB<1 || bufferMiB>64) throw new ArgumentException("buffer out of bounds");
        uint block=(uint)bufferMiB*1024*1024;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN; open existing file read-only.
        using(var file=CreateFile(path,0x80000000,1,IntPtr.Zero,3,0x28000000,IntPtr.Zero)) {
            if(file.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
            long size;
            if(!GetFileSizeEx(file,out size)) throw new Win32Exception(Marshal.GetLastWin32Error());
            if(size!=5368709120L || size%block!=0) throw new ArgumentException("expected aligned 5 GiB dataset");
            var buffer=VirtualAlloc(IntPtr.Zero,(UIntPtr)block,0x3000,4);
            if(buffer==IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
            try {
                var timer=Stopwatch.StartNew();
                for(long offset=0;offset<size;offset+=block) {
                    uint read;
                    if(!ReadFile(file,buffer,block,out read,IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error());
                    if(read!=block) throw new InvalidOperationException("short read");
                }
                timer.Stop(); return timer.Elapsed.TotalMilliseconds;
            } finally { VirtualFree(buffer,UIntPtr.Zero,0x8000); }
        }
    }
    public static string[] Ports(string pnpId) {
        var rows=new System.Collections.Generic.List<string>();
        string path=@"\\?\"+pnpId.Replace('\\','#')+"#{f18a0e88-c30c-11d0-8815-00a0c906bed8}";
        using(var hub=CreateFile(path,0,3,IntPtr.Zero,3,0,IntPtr.Zero)) {
            if(hub.IsInvalid) return new string[]{"hub open error="+Marshal.GetLastWin32Error()+" "+pnpId};
            for(int port=1;port<=32;port++) {
                byte[] info=new byte[4096]; Array.Copy(BitConverter.GetBytes(port),info,4);
                uint returned;
                if(!DeviceIoControl(hub,0x220448,info,(uint)info.Length,info,(uint)info.Length,out returned,IntPtr.Zero)) continue;
                if(BitConverter.ToUInt32(info,32)!=1) continue; // DeviceConnected
                byte[] v2=new byte[16];
                Array.Copy(BitConverter.GetBytes(port),v2,4);
                Array.Copy(BitConverter.GetBytes(16),0,v2,4,4);
                Array.Copy(BitConverter.GetBytes(4),0,v2,8,4); // USB 3 protocol
                bool ok=DeviceIoControl(hub,0x22045c,v2,16,v2,16,out returned,IntPtr.Zero);
                uint flags=ok?BitConverter.ToUInt32(v2,12):0;
                rows.Add(String.Format("hub={0} port={1} VID={2:X4} PID={3:X4} speed_legacy={4} v2_ok={5} v2_flags=0x{6:X} operating_SS={7} operating_SSPlus={8}",
                    pnpId,port,BitConverter.ToUInt16(info,12),BitConverter.ToUInt16(info,14),info[23],ok,flags,(flags&1)!=0,(flags&4)!=0));
            }
        }
        return rows.ToArray();
    }
}
'@
if ($Mode -eq 'usb') {
    Get-CimInstance Win32_PnPEntity -Filter "Service='USBHUB3' OR Service='USBHUB'" |
        ForEach-Object { [InWiMeStorage]::Ports($_.PNPDeviceID) }
    exit
}
if ($Repeats -lt 10) { throw 'at least 10 repetitions required' }
if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw 'dataset file missing' }
for($r=0;$r -lt $Repeats;$r++) {
    $before=@(Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk | Select-Object Name,DiskReadBytesPersec)
    $elapsed=[InWiMeStorage]::Read($Path,$BufferMiB)
    $after=@(Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk | Select-Object Name,DiskReadBytesPersec)
    [PSCustomObject]@{ repeat=$r; path=$Path; bytes=5368709120L; buffer_mib=$BufferMiB;
        read_ms=$elapsed; GiB_s=5000/$elapsed; before=$before; after=$after } | ConvertTo-Json -Compress -Depth 4
}
