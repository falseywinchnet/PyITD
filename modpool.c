//modpool.c copyright joshuah.rainstar@gmail.com 2025 do whatever u want with it just dont sue me
//the author takes zero libility for any damages issues or other unexpected outcomes
//you agree to these terms if you do not delete this file thanks
 
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <limits.h>  // for INT_MAX

#if defined(__linux__)
  // Linux futex
  #include <linux/futex.h>
  #include <sys/syscall.h>
  #include <sys/time.h>
  #include <unistd.h>
  static void soft_park(volatile int *addr, int expected) {
    syscall(SYS_futex, addr, FUTEX_WAIT, expected, NULL, NULL, 0);
  }
  static void soft_unpark(volatile int *addr) {
    syscall(SYS_futex, addr, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
  }

#elif defined(_WIN32)
  // Windows WaitOnAddress
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static void soft_park(volatile int *addr, int expected) {
    WaitOnAddress((PVOID)addr, &expected, sizeof(int), INFINITE);
  }
  static void soft_unpark(volatile int *addr) {
    WakeByAddressAll((PVOID)addr);
  }

#elif defined(__APPLE__)
  // macOS: use private ulock APIs (risky) or Mach semaphores; here we use ulock
  #include <sys/ulock.h>
  #include <unistd.h>
  static void soft_park(volatile int *addr, int expected) {
    // ULF_NO_ERRNO=0x00000002
    __ulock_wait(UL_UNFAIR_LOCK | 0x2, addr, expected, 0);
  }
  static void soft_unpark(volatile int *addr) {
    __ulock_wake(UL_UNFAIR_LOCK | 0x2, addr, 0);
  }

#else
  // Fallback: spin+yield
  #include <sched.h>
  static void soft_park(volatile int *addr, int expected) {
    while (*addr == expected) sched_yield();
  }
  static void soft_unpark(volatile int *addr) {
    // nothing—parkers will exit spin
  }
#endif

// Config
#define TASK_COUNT   100000
#define THREAD_COUNT 8
#define WORK_US      1000    // simulated task via usleep

// ─── Task Definition ───────────────────────────────────────────────────────────

typedef struct {
    void (*function)(void *);
    void *arg;
} threadpool_task_t;

static void do_work(void *arg) {
    (void)arg;
    usleep(WORK_US);
}

// ─── Soft-park Coop Pool Impl ─────────────────────────────────────────────────

struct mod_pool_t {
    pthread_t threads[THREAD_COUNT];
    // shared signal words
    atomic_int work_ready;    // 0=block, 1=go
    atomic_int done_signal;   // 0=wait, 1=done
    atomic_int shutdown;
    // task data
    atomic_int task_index;
    atomic_int task_count;
    atomic_int done_count;
    threadpool_task_t *tasks;
};

static void *mod_worker(void *arg) {
    mod_pool_t *M = (mod_pool_t*)arg;
    while (!atomic_load(&M->shutdown)) {
        // wait for main to signal work_ready==1
        while (atomic_load(&M->work_ready) == 0)
            soft_park((volatile int*)&M->work_ready, 0);
        // claim next task
        int i = atomic_fetch_add(&M->task_index, 1);
        if (i < atomic_load(&M->task_count)) {
            M->tasks[i].function(M->tasks[i].arg);
            // last to finish signals main
            if (atomic_fetch_add(&M->done_count, 1) + 1
                == atomic_load(&M->task_count)) {
                atomic_store(&M->done_signal, 1);
                soft_unpark((volatile int*)&M->done_signal);
            }
        } else {
            // batch done, park until next round
            while (atomic_load(&M->work_ready) == 1)
                sched_yield();
        }
    }
    return NULL;
}

mod_pool_t* modpool_create(void) {
    mod_pool_t *M = calloc(1, sizeof(*M));
    atomic_store(&M->work_ready, 0);
    atomic_store(&M->done_signal, 0);
    atomic_store(&M->shutdown, 0);
    for (int i = 0; i < THREAD_COUNT; i++)
        pthread_create(&M->threads[i], NULL, mod_worker, M);
    return M;
}

void modpool_run(mod_pool_t *M, threadpool_task_t *tasks, int n) {
    M->tasks = tasks;
    // reset batch signals
    atomic_store(&M->task_index, 0);
    atomic_store(&M->done_count, 0);
    atomic_store(&M->task_count, n);
    atomic_store(&M->done_signal, 0);
    // signal workers to start
    atomic_store(&M->work_ready, 1);
    soft_unpark((volatile int*)&M->work_ready);
    // main parks until done_signal==1
    while (atomic_load(&M->done_signal) == 0)
        soft_park((volatile int*)&M->done_signal, 0);
    // reset for next batch
    atomic_store(&M->work_ready, 0);
}

void modpool_destroy(mod_pool_t *M) {
    atomic_store(&M->shutdown, 1);
    // wake all workers so they can exit
    atomic_store(&M->work_ready, 1);
    soft_unpark((volatile int*)&M->work_ready);
    for (int i = 0; i < THREAD_COUNT; i++)
        pthread_join(M->threads[i], NULL);
    free(M);
}

// ─── Benchmark Harness ────────────────────────────────────────────────────────

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(const char *label,
                  void (*run_fn)(threadpool_task_t*, int)) {
    threadpool_task_t *tasks =
      calloc(TASK_COUNT, sizeof(threadpool_task_t));
    for (int i = 0; i < TASK_COUNT; i++) {
        tasks[i].function = do_work;
        tasks[i].arg      = NULL;
    }
    double start = now_sec();
    run_fn(tasks, TASK_COUNT);
    double end   = now_sec();
    double dur   = end - start;
    printf("%-25s: %.3f sec  |  %.0f tasks/sec\n",
           label, dur, TASK_COUNT/dur);
    free(tasks);
}

static void run_original(threadpool_task_t *t, int n) {
    threadpool_t *P = threadpool_create(THREAD_COUNT);
    threadpool_run(P, t, n);
    threadpool_destroy(P);
}

static void run_modpool(threadpool_task_t *t, int n) {
    static mod_pool_t *M = NULL;
    if (!M) M = modpool_create();
    modpool_run(M, t, n);
}