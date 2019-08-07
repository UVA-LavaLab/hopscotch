#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <string>
#include <cmath>
#include <iosfwd>
#include <map>
#include "pin.H"

using namespace std;

#define MAX_THREAD_NUM      1024

// Name of the function to profile
KNOB<string> knobFunctionName(KNOB_MODE_WRITEONCE, "pintool", "func", "main", "Function to be profiled.");

// Max log items per thread
KNOB<UINT64> knobMaxLog(KNOB_MODE_WRITEONCE, "pintool", "lim", "100000", "Max number of read/writes to log per thread.");

// Output file name
KNOB<string> knobOutFile(KNOB_MODE_WRITEONCE, "pintool", "out", "log_map.csv", "Output file name.");

// Log level--
// 1: Log only the specified function
// N: Log upto N function calls from the specified functions. Set to a big number to deep logging
KNOB<UINT64> knobLogLevel(KNOB_MODE_WRITEONCE, "pintool", "level", "1000", "Log level. 1: Without deep logging.");

typedef struct{
    UINT64 rtnEntryCnt = 0;
    UINT8 _pad[64 - 8];
} ThreadData;

typedef struct{
    ADDRINT ea;         // address
    UINT8 type;         // 'R' or 'W'
    UINT8 _pad[64 - sizeof(ADDRINT) - 8];
} MemInfo;

MemInfo* info = NULL;
ThreadData tdata[MAX_THREAD_NUM];
UINT64 buf_log_lim = 0;
UINT64 buf_log_cnt = 0;
UINT64 log_level = 0;
string rtn_name;
PIN_LOCK lock;



INT32 Usage() {
    cerr << "This tool profiles a function\'s memory access pattern." << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}

VOID record(THREADID tid, ADDRINT ea, UINT8 type){
    UINT64 entCnt = tdata[tid].rtnEntryCnt;
    if(entCnt > 0 && entCnt <= log_level){
        UINT64 idx;
        PIN_GetLock(&lock, tid + 1);
        idx = buf_log_cnt++;
        PIN_ReleaseLock(&lock);
        if(idx < buf_log_lim){
            info[idx].ea = ea;
            info[idx].type = type;
        } else {
            PIN_ExitApplication(0);
        }
    }
}

// Print a memory read record
VOID RecordMemRead(THREADID tid, ADDRINT ea) {
    record(tid, ea, '0');
}

// Print a memory write record
VOID RecordMemWrite(THREADID tid, ADDRINT ea) {
    record(tid, ea, '1');
}
    
VOID Instruction(INS ins, VOID *v){
    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
        if (INS_MemoryOperandIsRead(ins, memOp)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead, IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp, IARG_END);
        }

        if (INS_MemoryOperandIsWritten(ins, memOp)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite, IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp, IARG_END);
        }
    }
}

VOID RtnEntry(THREADID tid){
    tdata[tid].rtnEntryCnt++;
}

VOID RtnLeave(THREADID tid){
    tdata[tid].rtnEntryCnt--;
}

VOID ImgCallback(IMG img, VOID* arg){
    if (!IMG_IsMainExecutable(img))
        return;
    
	for( SEC sec= IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)){
        if(SEC_Name(sec) == ".text"){
            for(RTN rtn= SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)){
                string name = PIN_UndecorateSymbolName(RTN_Name(rtn), UNDECORATION_NAME_ONLY);
                if(name.find(rtn_name) != string::npos){
					cout << "-------------------Found " << rtn_name << "---------------------" << endl;
					RTN_Open(rtn);
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)RtnEntry, IARG_THREAD_ID, IARG_END);
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)RtnLeave, IARG_THREAD_ID, IARG_END);
					RTN_Close(rtn);
					return;
				}
            }
        }
    }
	cout << "ERROR: Unable to find " << rtn_name << "." << endl;
    PIN_ExitProcess(11);
}

VOID Fini(INT32 code, VOID *v) {
    string out_file = knobOutFile.Value();
    ofstream log(out_file.c_str());
    if(!log.is_open()) {
        cerr << "Cannot open log file." << endl;
    }

    //Write headers for csv files
    log << "R0_W1,Addr\n";
    
    if(buf_log_cnt > buf_log_lim){
        buf_log_cnt = buf_log_lim;
    }
    
    for(UINT64 i=0; i < buf_log_cnt; i++){
        log << info[i].type << "," << info[i].ea << "\n";
    }
    
    log.close();
    delete [] info;
}

int main(int argc, char *argv[]) {
    if(PIN_Init(argc,argv)) {
        return Usage();
    }
    
    PIN_InitLock(&lock);
    // Initialize symbol processing
    PIN_InitSymbolsAlt(IFUNC_SYMBOLS);
    
    buf_log_lim = knobMaxLog.Value();
    log_level = knobLogLevel.Value();
    info = new MemInfo[buf_log_lim];
    rtn_name = knobFunctionName.Value();
    cout << "Profiling for function: " << rtn_name << endl;
    
    IMG_AddInstrumentFunction(ImgCallback, NULL);
    INS_AddInstrumentFunction(Instruction, NULL);
    PIN_AddFiniFunction(Fini, NULL);
    
    // Start the program, never returns
    PIN_StartProgram();

    return 0;
}


