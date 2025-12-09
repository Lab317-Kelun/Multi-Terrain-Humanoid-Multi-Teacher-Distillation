file(REMOVE_RECURSE
  "lib/liblcm_types_lib.a"
  "lib/liblcm_types_lib.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/lcm_types_lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
