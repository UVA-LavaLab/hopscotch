/*******************************************************************************
 *
 * File: MAPProfiler.cpp
 * Description: Pin tool for tracing memory accesses.
 * 
 * Author:  Alif Ahmed
 * Email:   alifahmed@virginia.edu
 * Updated: Aug 06, 2019
 *
 ******************************************************************************/
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


/*******************************************************************************
 * Knobs for configuring the instrumentation
 ******************************************************************************/
// Name of the function to profile
KNOB<string> knobFunctionName(KNOB_MODE_WRITEONCE, "pintool", "func", "main",
        "Function to be profiled.");

// Max log items per thread
KNOB<UINT64> knobMaxLog(KNOB_MODE_WRITEONCE, "pintool", "lim", "1000000", 
        "Max number of read/writes to log per thread.");

// Output file name
KNOB<string> knobOutFile(KNOB_MODE_WRITEONCE, "pintool", "out", "mem_trace.csv",
        "Output file name.");

// Max threads
KNOB<UINT64> knobMaxThreads(KNOB_MODE_WRITEONCE, "pintool", "threads", "10000",
        "Upper limit of the number of threads that can be used by the program \
        being profiled.");

// Stack based access logging (1: enable, 0: disable)
KNOB<bool> knobStack(KNOB_MODE_WRITEONCE, "pintool", "stack", "0", "Stack based access logging \
        [1: enable, 0: disable (default)].");

// Instruction pointer relative access logging (1: enable, 0: disable)
KNOB<bool> knobIP(KNOB_MODE_WRITEONCE, "pintool", "ip", "1", "IP relative access logging \
        [1: enable (default), 0: disable].");

// Read logging (1: enable, 0: disable)
KNOB<bool> knobRead(KNOB_MODE_WRITEONCE, "pintool", "read", "1", "Read logging \
        [1: enable (default), 0: disable].");

// Write logging (1: enable, 0: disable)
KNOB<bool> knobWrite(KNOB_MODE_WRITEONCE, "pintool", "write", "1", "Write \
        logging [1: enable (default), 0: disable].");



/*******************************************************************************
 * Structs
 ******************************************************************************/
#define LINE_SIZE   64

// Structure for keeping thread specific data. Padded to LINE_SIZE for avoiding
// false sharing.
typedef struct{
    // Tracks if the thread is inside the requested functions. Greater than 0
    // means it is.
    UINT64 rtnEntryCnt;

    // padding
    UINT8 _padding[LINE_SIZE - sizeof(UINT64)];
} ThreadData;


// Keeps information for each memory access. Padded to LINE_SIZE for avoiding
// false sharing.
typedef struct{
    // Effective virtual address
    ADDRINT ea;

    // Type of access. 'R' for read and 'W' for write.
    UINT8 type;

    // padding
    UINT8 _padding[LINE_SIZE - sizeof(ADDRINT) - sizeof(UINT8)];
} MemInfo;



/*******************************************************************************
 * Globals
 ******************************************************************************/
// Array that contains all the memory accesses.
MemInfo* info;

// Array that keeps thread specific data for all threads.
ThreadData* tdata;

// Limit of buffer entry. Maps to knobMaxLog.
UINT64 buf_log_lim;

// Current count of log entry.
UINT64 buf_log_cnt;

// Name of the function that is being profiled. Maps to knobFunctionName.
string rtn_name;

// Lock for synchronizing thread access to 'info' array.
PIN_LOCK lock;

// Read logging is enabled if true. Maps to knobRead.
bool read_log_en = true;

// Write logging is enabled if true. Maps to knobWrite.
bool write_log_en = true;


/*******************************************************************************
 * Functions
 ******************************************************************************/

/**
 * Prints usage message. This function is called if any argument is invalid.
 */
INT32 Usage() {
    cerr << "This tool profiles a function\'s memory access pattern." << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}


/**
 * Records a read or write access to 'info' array.
 */
inline VOID record(THREADID tid, ADDRINT ea, UINT8 type){
    // First check if thread is inside the function beign profiled.
    UINT64 entCnt = tdata[tid].rtnEntryCnt;
    if(entCnt > 0){
        // Inside the function. Atomically update the array index.
        UINT64 idx;
        PIN_GetLock(&lock, tid + 1);
        idx = buf_log_cnt++;
        PIN_ReleaseLock(&lock);

        // Check if log limit reached.
        if(idx < buf_log_lim){
            // Record entry.
            info[idx].ea = ea;
            info[idx].type = type;
        } else {
            // Log limit reached. Exit.
            // Intenally calls FINI function before quitting.
            PIN_ExitApplication(0);
        }
    }
}


/**
 * Function for recording read access.
 */
VOID RecordMemRead(THREADID tid, ADDRINT ea) {
    record(tid, ea, '0');
}


/**
 * Function for recording write access.
 */
VOID RecordMemWrite(THREADID tid, ADDRINT ea) {
    record(tid, ea, '1');
}


/**
 * Instruments instructions having read or write accesses.
 */
VOID Instruction(INS ins, VOID *v){
    if(!knobStack.Value()){
        if(INS_IsStackRead(ins) || INS_IsStackWrite(ins)){
            return;
        }
    }

    if(!knobIP.Value()){
        if(INS_IsIpRelRead(ins) || INS_IsIpRelWrite(ins)){
            return;
        }
    }

    // Get the memory operand count of the current instruction.
    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
        if (INS_MemoryOperandIsRead(ins, memOp) && read_log_en) {
            // Operand is read by this instruction.
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead, IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp, IARG_END);
        }

        else if (INS_MemoryOperandIsWritten(ins, memOp) && write_log_en) {
            // Operand is written by this instruction.
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite, IARG_THREAD_ID, IARG_MEMORYOP_EA, memOp, IARG_END);
        }
    }
}


/**
 * Keeps track of the function entry.
 */
VOID RtnEntry(THREADID tid){
    tdata[tid].rtnEntryCnt++;
}


/**
 * Keeps track of the function exit.
 */
VOID RtnLeave(THREADID tid){
    tdata[tid].rtnEntryCnt--;
}


/**
 * Finds and instruments the requested routine.
 */
VOID ImgCallback(IMG img, VOID* arg){
    if (!IMG_IsMainExecutable(img))
        return;
    
    int match_count = 0;
    cout << "Tagging functions with \"" << rtn_name << "\"..." << endl;

    //First try for exact match of function
    for( SEC sec= IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)){
        if(SEC_Name(sec) == ".text"){
            for(RTN rtn= SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)){
                string name = PIN_UndecorateSymbolName(RTN_Name(rtn), UNDECORATION_NAME_ONLY);
                // add suffix for openmp functions
                string rtn_name_omp = rtn_name + "._omp_fn.";   

                // Try exact name match
                if((name == rtn_name) || (name.find(rtn_name_omp) != string::npos)){
                    // Match found!
                    match_count++;
                    cout << "    Tagged function \"" << name << "\"" << endl;

                    // Instrument function entry and exit
                    RTN_Open(rtn);
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)RtnEntry, IARG_THREAD_ID, IARG_END);
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)RtnLeave, IARG_THREAD_ID, IARG_END);
                    RTN_Close(rtn);
                }
            }
        }
    }
    if(match_count) return;
    
    //Exact match not found. Try to find a function containing the given function name.
    cout << "Exact match not found! Tagging all functions containing \"" << rtn_name << "\"..." << endl;
    for( SEC sec= IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec)){
        if(SEC_Name(sec) == ".text"){
            for(RTN rtn= SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn)){
                string name = PIN_UndecorateSymbolName(RTN_Name(rtn), UNDECORATION_NAME_ONLY);

                // Check if the current routine contains the requested routine name
                if(name.find(rtn_name) != string::npos){
                    // Match found!
                    match_count++;
                    cout << "    Tagged function \"" << name << "\"" << endl;

                    // Instrument function entry and exit
                    RTN_Open(rtn);
                    RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)RtnEntry, IARG_THREAD_ID, IARG_END);
                    RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)RtnLeave, IARG_THREAD_ID, IARG_END);
                    RTN_Close(rtn);
                }
            }
        }
    }

    //Not found
    if(!match_count){
        cout << "Unable to find any function containing \"" << rtn_name << "\"... Quitting..." << endl;
        PIN_ExitProcess(11);
    }
}


/**
 * This function is called when exiting Pin. Writes the entries into a log file.
 */
VOID Fini(INT32 code, VOID *v) {
    // Open log file.
    string out_file = knobOutFile.Value();
    ofstream log(out_file.c_str());
    if(!log.is_open()) {
        cerr << "Cannot open log file:" << out_file << endl;
        PIN_ExitProcess(11);
    }
    else{
        cout << "Writing trace to " << out_file << "... " ;
    }

    //Write headers for csv files
    log << "R0_W1,Addr\n";
    
    // write log
    if(buf_log_cnt > buf_log_lim){
        buf_log_cnt = buf_log_lim;
    }
    
    for(UINT64 i=0; i < buf_log_cnt; i++){
        log << info[i].type << "," << info[i].ea << "\n";
    }
    
    // cleanup tasks
    log.close();
    delete [] info;
    delete [] tdata;

    cout << "Done" << endl;
}


/**
 * Entry point
 */
int main(int argc, char *argv[]) {
    if(PIN_Init(argc,argv)) {
        return Usage();
    }

    // Check if MemInfo and ThreadData structures are properly padded.
    // Padding is used to avoid false sharing.    
    assert(sizeof(MemInfo) == LINE_SIZE);
    assert(sizeof(ThreadData) == LINE_SIZE);

    // Initializations
    PIN_InitLock(&lock);
    PIN_InitSymbolsAlt(IFUNC_SYMBOLS);
    
    buf_log_lim = knobMaxLog.Value();
    info = new MemInfo[buf_log_lim];
    UINT64 max_threads = knobMaxThreads.Value();
    tdata = new ThreadData[max_threads];
    for(UINT64 i = 0; i < max_threads; ++i){
        tdata[i].rtnEntryCnt = 0;
    }
    rtn_name = knobFunctionName.Value();
    read_log_en = knobRead.Value();
    write_log_en = knobWrite.Value();

    IMG_AddInstrumentFunction(ImgCallback, NULL);
    INS_AddInstrumentFunction(Instruction, NULL);
    PIN_AddFiniFunction(Fini, NULL);
    
    // Start the program, never returns
    PIN_StartProgram();

    return 0;
}


