__kernel void find_sum(__global const float* input, __global float* output, __local float* local_cache) {
	
	/*Grab relevant thread & work item info */
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy values into local memory
	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are caught up before processing anything

	for (int i = local_size/2; i > 0; i /= 2) {
		if(local_id < i){
			local_cache[local_id] += local_cache[local_id + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are done before incrementing the loop
	}
	if (!local_id) {
		//Get the value to add
		float add_val = local_cache[local_id];
		while (add_val != 0.0) {
			/*Using atomics avoids threads accessing memory at the same time */
			//Atmomic_xchg lets floats be used in atomic functions
			float old_val = atomic_xchg(&output[0], 0.0); //return the value in output [0] and change it for 0.0
			add_val = atomic_xchg(&output[0], old_val + add_val); //change the value in output[0] for the original output[0] summed with the local value
		}
	}
}


__kernel void find_max(__global const float* input, __global float* output, __local float* local_cache) {

	/*Grab relevant thread & work item info */
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy values into local memory
	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are caught up before processing anything

	for (int i = local_size / 2; i > 0; i /= 2) {
		if (local_id < i) {
			local_cache[local_id] = max(local_cache[local_id + i], local_cache[local_id]);
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are done before incrementing the loop
	}
	if (!local_id) {
		float new_val = local_cache[local_id];
		while (new_val != 0.0) {
			/*Using atomics avoids threads accessing memory at the same time */
			//Atmomic_xchg lets floats be used in atomic functions
			float old_val = atomic_xchg(&output[0], 0.0); //return the value in output [0] and change it for 0.0
			new_val = atomic_xchg(&output[0], max(old_val, new_val)); //change output[0] for the maximum of the original output[0] value and the local value
		}
	}
}


__kernel void find_min(__global const float* input, __global float* output, __local float* local_cache) {

	/*Grab relevant thread & work item info */
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy values into local memory
	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are caught up before processing anything

	for (int i = local_size / 2; i > 0; i /= 2) {
		if (local_id < i) {
			local_cache[local_id] = min(local_cache[local_id + i], local_cache[local_id]);
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are done before incrementing the loop
	}
	if (!local_id) {
		float new_val = local_cache[local_id];
		while (new_val != 0.0) {
			/*Using atomics avoids threads accessing memory at the same time */
			//Atmomic_xchg lets floats be used in atomic functions
			float old_val = atomic_xchg(&output[0], 0.0); //return the value in output [0] and change it for 0.0
			new_val = atomic_xchg(&output[0], min(old_val, new_val)); //change output[0] for the minimum of the original output[0] value and the local value
		}
	}
}

__kernel void variance(__global const float* input, __global float* output, __local float* local_cache, const float mean) {
	
	/*Grab relevant thread & work item info */
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy values into local memory
	local_cache[local_id] = pow((mean-input[id]),2); //Instead of passing in raw input values we're only interested in values that are the square difference of the mean and that value 
	barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are caught up before processing anything

	for (int i = local_size / 2; i > 0; i /= 2) {
		if (local_id < i) {
			local_cache[local_id] += local_cache[local_id + i]; //Simple summation
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are done before incrementing the loop
	}
	if (!local_id) {
		float add_val = local_cache[local_id];
		while (add_val != 0.0) {
			/*Using atomics avoids threads accessing memory at the same time */
			//Atmomic_xchg lets floats be used in atomic functions
			float old_val = atomic_xchg(&output[0], 0.0); //return the value in output [0] and change it for 0.0
			add_val = atomic_xchg(&output[0], old_val + add_val); //Change output[0] for the sum of the original value and the local value
		}
	}
}

__kernel void histogram(__global float* input, __global int* hist, __local int* local_cache, const int bin_size, float min_val, float max_val) {
	
	/*Grab relevant thread & work item info */
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//Copy input into local memory
	local_cache[local_id] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE); //Make sure all threads are caught up before processing anything

	int nr_of_bins = (max_val - min_val) / bin_size; //Find the number of bins

	int bin_index = (int)(input[id] - min_val) / bin_size;
	if(bin_index > 0 && bin_index < nr_of_bins){ //Make sure the calculated bin index falls in the range of the bins (avoid out of resource errors)
		atomic_inc(&hist[bin_index]); //Use atomic inc to avoid cconflicts, this works as we are using integer values
	}
}