conf_pos0={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}}
}
conf_pos2={"2":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[2,2]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}}
}
conf_pos5={"5":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":64,
                                           "read_head_num":3,
                                           "write_head_num":3}}
}


conf_pos02={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}},
            "2":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[2,2]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}}
}
conf_pos05={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}},
            "5":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":64,
                                           "read_head_num":3,
                                           "write_head_num":3}}
}

conf_pos025 = {}
for d in [conf_pos0, conf_pos2, conf_pos5]:
    conf_pos025.update(d)

CONF_POS_LIST=[conf_pos0, conf_pos2, conf_pos5, conf_pos02, conf_pos05, conf_pos025]


conf_mem128_vec128={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":128,
                                           "memory_vector_dim":128,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}}}
conf_mem64_vec512={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":512,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}}}
conf_r1_w1={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":1,
                                           "write_head_num":1}}}
conf_r5_w5={"0":{"enc_dec_param":{"num_filters":64,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":5,
                                           "write_head_num":5}}}

CONF_MEM_LIST = [conf_mem128_vec128, conf_mem64_vec512, conf_r1_w1, conf_r5_w5]