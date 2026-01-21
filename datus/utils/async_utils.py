# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
在各种上下文中运行异步代码的健壮异步工具
优雅地处理同步和异步环境
"""

import asyncio
import logging
import sys
import threading
import weakref
from typing import Any, Coroutine, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 线程本地存储，用于跟踪嵌套调用和循环所有权
_local = threading.local()

# 跟踪此模块创建的所有循环以便清理
_created_loops = weakref.WeakSet()


def setup_windows_policy():
    """设置Windows特定的事件循环策略以获得更好的兼容性

    ProactorEventLoop在子进程管道方面有局限性
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def is_event_loop_running() -> bool:
    """检查事件循环当前是否正在运行

    这比仅检查get_running_loop()更健壮

    Returns:
        bool: 如果事件循环正在运行返回True，否则返回False
    """
    try:
        loop = asyncio.get_running_loop()
        # 再次检查循环确实在运行
        return loop is not None and loop.is_running() and not loop.is_closed()
    except RuntimeError:
        # 当前上下文中没有运行的循环
        return False


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """获取当前事件循环或在必要时创建一个新的

    Returns:
        asyncio.AbstractEventLoop: 事件循环实例

    Note:
        此函数不处理循环已在运行的情况。
        对于这种情况，请使用`run_async`。
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _created_loops.add(loop)
        return loop
    except RuntimeError:
        # 当前线程中没有事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _created_loops.add(loop)
        return loop


def run_async(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """智能异步协程运行器，适用于任何上下文

    此函数可以从以下环境调用：
    - 同步代码(将创建和管理事件循环)
    - 异步函数内部(将使用线程池)
    - 带有或不带有事件循环的线程

    Args:
        coro: 要运行的协程
        timeout: 可选的超时时间(秒)

    Returns:
        T: 协程的结果

    Raises:
        asyncio.TimeoutError: 如果指定了超时并超出
        Exception: 协程抛出的任何异常
    """
    # 检查嵌套调用以防止死锁
    if hasattr(_local, "in_run_async") and _local.in_run_async:
        logger.warning("检测到嵌套run_async，使用线程池以避免死锁")
        return _run_in_thread(coro, timeout)

    # 检查我们是否在异步上下文中
    if is_event_loop_running():
        # 我们已经在异步上下文中，使用线程池
        logger.debug("检测到运行中的事件循环，使用线程池执行器")
        return _run_in_thread(coro, timeout)
    else:
        # 没有运行的循环，我们可以安全地创建和使用一个
        logger.debug("没有运行的事件循环，创建新的")
        _local.in_run_async = True
        try:
            return _run_in_new_loop(coro, timeout)
        finally:
            _local.in_run_async = False


def _run_in_new_loop(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """在新的事件循环中运行协程，并进行改进的清理

    Args:
        coro: 要运行的协程
        timeout: 可选的超时时间(秒)

    Returns:
        T: 协程的结果
    """
    loop = None
    original_loop = None

    try:
        # 存储当前线程的事件循环(如果有的话)
        try:
            original_loop = asyncio.get_event_loop()
            if original_loop and original_loop.is_closed():
                original_loop = None
        except RuntimeError:
            original_loop = None

        # 创建一个新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _created_loops.add(loop)

        # 如果指定了超时，则用超时包装
        if timeout is not None:

            async def with_timeout():
                return await asyncio.wait_for(coro, timeout)

            task_to_run = with_timeout()
        else:
            task_to_run = coro

        # 运行协程
        return loop.run_until_complete(task_to_run)

    finally:
        # 彻底清理
        if loop is not None:
            try:
                # 取消任何剩余的任务
                pending = asyncio.all_tasks(loop) if hasattr(asyncio, "all_tasks") else asyncio.Task.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # 短暂运行循环以处理取消
                if pending:
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=1.0)
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

                # 绝对确保循环已停止
                loop.call_soon(loop.stop)
                if loop.is_running():
                    loop.run_until_complete(asyncio.sleep(0))
                    loop.stop()

                # 关闭循环
                loop.close()

            except Exception as e:
                logger.warning(f"循环清理期间出错: {e}")

        # 恢复或清除此线程的事件循环
        if original_loop is not None and not original_loop.is_closed():
            asyncio.set_event_loop(original_loop)
        else:
            # 重要：显式设置为None以清除任何循环引用
            asyncio.set_event_loop(None)

        logger.debug(f"循环清理完成，已恢复: {original_loop}")


def _run_in_thread(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """在带有自己事件循环的单独线程中运行协程

    Args:
        coro: 要运行的协程
        timeout: 可选的超时时间(秒)

    Returns:
        T: 协程的结果

    Raises:
        Exception: 协程抛出的任何异常
    """
    result_container: Dict[str, Any] = {"result": None, "exception": None, "loop": None}
    stop_event = threading.Event()

    def thread_target():
        """线程的目标函数"""
        loop = None
        try:
            # 为此线程创建一个新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_container["loop"] = loop

            async def run_with_stop_check():
                task = asyncio.create_task(coro)

                async def check_stop():
                    while not stop_event.is_set():
                        await asyncio.sleep(0.1)
                    task.cancel()

                stop_task = asyncio.create_task(check_stop())

                try:
                    if timeout:
                        return await asyncio.wait_for(task, timeout)
                    else:
                        return await task
                finally:
                    stop_task.cancel()
                    try:
                        await stop_task
                    except asyncio.CancelledError:
                        pass

            result = loop.run_until_complete(run_with_stop_check())
            result_container["result"] = result

        except Exception as e:
            result_container["exception"] = e
        finally:
            # 清理线程的循环
            if loop is not None:
                try:
                    # 取消任何剩余的任务
                    pending = asyncio.all_tasks(loop) if hasattr(asyncio, "all_tasks") else asyncio.Task.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        try:
                            loop.run_until_complete(
                                asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=0.5)
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass

                    if loop.is_running():
                        loop.stop()
                    loop.close()
                except Exception as e:
                    logger.warning(f"关闭线程循环时出错: {e}")
                finally:
                    # 始终清除此线程的循环
                    asyncio.set_event_loop(None)

    # 创建并运行线程
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    # 检查线程是否仍在运行(超时情况)
    if thread.is_alive():
        stop_event.set()
        if result_container["loop"]:
            try:
                result_container["loop"].call_soon_threadsafe(result_container["loop"].stop)
            except Exception:
                pass

        thread.join(timeout=0.5)

        if thread.is_alive():
            logger.error("线程未能优雅地停止")

        raise asyncio.TimeoutError(f"协程执行超出{timeout}秒的超时时间")

    # 检查异常
    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["result"]
