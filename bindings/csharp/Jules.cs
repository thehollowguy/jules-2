using System;
using System.Runtime.InteropServices;

public static class JulesNative
{
    private const string Dll = "jules";

    public enum JulesError : uint
    {
        Success = 0,
        InvalidArg = 1,
        RuntimeError = 2,
        OutOfMemory = 3,
        NotFound = 4,
        UnknownError = 255,
    }

    [DllImport(Dll)] public static extern IntPtr jules_init();
    [DllImport(Dll)] public static extern void jules_destroy(IntPtr ctx);
    [DllImport(Dll)] public static extern uint jules_version();
    [DllImport(Dll)] public static extern IntPtr jules_error_string(JulesError code);

    [DllImport(Dll, CharSet = CharSet.Ansi)]
    public static extern JulesError jules_run_file_ffi(string path);

    [DllImport(Dll, CharSet = CharSet.Ansi)]
    public static extern JulesError jules_check_code_ffi(string source);

    public static string ErrorToString(JulesError err)
        => Marshal.PtrToStringAnsi(jules_error_string(err)) ?? "Unknown";
}
