#pragma once

#include "Population.h"

#define NOMINMAX
#include <windows.h>

#define RECURSIVE_NODES_API __declspec(dllexport)

#ifdef DRAWING
#include "Drawer.h"
#endif 


extern "C" {

	BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

	
	
	RECURSIVE_NODES_API void prepare_network(Network* n) {
		n->createPhenotype();
		n->preTrialReset();
	}
	
	RECURSIVE_NODES_API void get_actions(Network* n, float* observations, float* actions) {
		std::vector<float> observationsV(observations, observations + n->inputSize);
		n->step(observationsV);
		float* actionsTemp = n->getOutput();
		std::copy(actionsTemp, actionsTemp+n->outputSize, actions);
	}

	RECURSIVE_NODES_API Network* load_existing_network(const char* path) {
		std::ifstream is(path, std::ios::binary);
		if (!is.is_open()) {
			std::string path_s(path);
			std::cout << "SPECIFIED NETWORK WAS NOT FOUND. PATH WAS : \n" 
				<< path_s << std::endl;
			return nullptr;
		}
		return new Network(is);
	}

	RECURSIVE_NODES_API int get_observations_size(Network* n) {
		return n->inputSize;
	}

	RECURSIVE_NODES_API int get_actions_size(Network* n) {
		return n->;
	}
}