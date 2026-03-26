#pragma once

#include <cstdint>

extern "C" {
    typedef struct JulesContext JulesContext;
    typedef struct JulesTensor JulesTensor;

    enum JulesError {
        JulesError_Success = 0,
        JulesError_InvalidArg = 1,
        JulesError_RuntimeError = 2,
        JulesError_OutOfMemory = 3,
        JulesError_NotFound = 4,
        JulesError_UnknownError = 255,
    };

    JulesContext* jules_init();
    void jules_destroy(JulesContext*);

    uint32_t jules_version();
    const char* jules_error_string(JulesError);

    JulesError jules_run_file_ffi(const char* path);
    JulesError jules_check_code_ffi(const char* source);

    JulesTensor* jules_tensor_create(const size_t* shape, size_t shape_len);
    void jules_tensor_destroy(JulesTensor* tensor);
    const float* jules_tensor_data(const JulesTensor* tensor);
    const size_t* jules_tensor_shape(const JulesTensor* tensor);
    size_t jules_tensor_shape_len(const JulesTensor* tensor);
    size_t jules_tensor_numel(const JulesTensor* tensor);
}
