function(log_info TEXT)
    message(STATUS ${TEXT})
endfunction()

function(log_warning TEXT)
    message(WARNING ${TEXT})
endfunction()

function(log_fatal TEXT)
    message(FATAL_ERROR ${TEXT})
endfunction()
